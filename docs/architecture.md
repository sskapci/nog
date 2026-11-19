# NOG architecture

How the pieces fit, from the whole system down to one knowledge unit (KU). Code: `mxn/neuron/` (KU, host, gateway);
numbers and tests in [results.md](results.md), method in [paper.md](paper.md).

## 1. System overview

```mermaid
flowchart TB
  Q(["Question"]) --> GW
  subgraph S["Supplements: hold no KU knowledge"]
    GW["Gateway / driver"]
    PL["Planner"]
    LE["Language expert (frozen encoder)"]
    RS["Resultant"]
  end
  GW --> PL
  GW --> LE
  subgraph H1["Host 1"]
    K1(("KU"))
    K2(("KU"))
    I1[("Own title index, slot-value cache, cold store")]
  end
  subgraph H2["Host 2"]
    K3(("KU"))
    K4(("KU"))
    I2[("Own title index, slot-value cache, cold store")]
  end
  GW -- "RESOLVE broadcast" --> H1
  GW -- "RESOLVE broadcast" --> H2
  GW -- "TASKS, first hop" --> K1
  K2 -- "next hop via own pointer" --> K3
  K1 -- "finding" --> RS
  K3 -- "finding" --> RS
  RS --> A(["Answer + source KUs, or abstain"])
```

Each host holds only its own KUs, placed by rendezvous hashing over the host list, and builds its own index and caches
for them; no index or directory spans hosts. The supplements turn words into signals (planner, language expert) and
apply exact operations to the KUs' findings (resultant), but never read KU knowledge. After the first hop the gateway
does not orchestrate: KUs forward the next hop themselves.

## 2. Question flow

```mermaid
sequenceDiagram
  participant U as User
  participant G as Gateway
  participant P as Planner + language expert
  participant H as All hosts
  participant A as KU A
  participant B as KU B
  participant R as Resultant
  U->>G: question
  G->>P: question
  P-->>G: plan (template, entity span, relation per hop) and signals
  G->>H: RESOLVE broadcast (entity span)
  H-->>G: candidates, each KU scores itself
  G->>A: task, hop 1 (query signal)
  A->>A: own forward pass: fire strength f and a slot
  A->>B: forwards hop 2 via its own pointer
  A-->>R: finding (f, answer words)
  B-->>R: finding (f, answer words)
  R-->>U: answer + source KUs, or abstain
```

The planner breaks the question into a chain of relations; the language expert encodes the entity span and the
question into signals. The KU that wins RESOLVE answers the first relation with its own network; if its answer names
another KU, it sends the next hop to that KU along its own pointer (or a RESOLVE broadcast when it has none). The
resultant combines the findings with exact operations and abstains when a chain is incomplete or a KU says
"don't know" (f below its threshold).

## 3. Inside one KU

```mermaid
flowchart LR
  subgraph T["Training (once, alone)"]
    DOC["Own document"] --> EX["Cloze examples from own sentences"]
    EX --> ENC["Language expert encodes them"]
    ENC --> NET
    DOC --> SL["Slot table: names and values in its text"]
  end
  subgraph KU["One KU after training (text deleted)"]
    NET["Own network: signal + relation, one hidden layer, slots + don't-know"]
    SL2["Slot table"]
    PT["Pointers: name slot to target KU"]
    TV["Own title vector"]
  end
  SIG(["Query signal + relation"]) --> NET
  NET --> OUT{"f >= threshold?"}
  OUT -- "yes" --> ANS(["Slot value, or forward along pointer"])
  OUT -- "no" --> DK(["Don't know"])
  SL --> SL2
```

A KU is one small network (input: query signal plus relation; one hidden layer; outputs: one per own slot plus
"don't know"), trained only on its own document: about 35–39k parameters (2Wiki and corpus A headline runs). Its slot table holds
the names and values found in its own text; name slots that match another KU's title become its pointers, found by its
own RESOLVE broadcast. After training the text is deleted and an audit checks that no sentence remains in memory;
answers are identical with and without the text.

## 4. Failover: hot copies, cold copies and takeover

```mermaid
flowchart TB
  P["Probes: 3 misses confirm a dead host"] --> D{"Copies kept?"}
  D -- "R = 2 hot copies" --> HOT["Surviving holder sends one copy to the host that completes R holders (re-replication)"]
  D -- "R = 1 cold copies" --> COLD["Survivors load the dead host's cold copies they now own"]
  HOT --> M["MEMBERS add / drop / done over the survivors"]
  COLD --> M
  M --> RA["Gateway re-asks incomplete questions (attempt number, older findings ignored)"]
  RA --> SAME(["Answers identical to before the failure"])
  subgraph G["Cold copies by generation"]
    W["write: each KU's package to its successor host's disk, not loaded"] --> C["commit: copies of older generations deleted"]
  end
  G -.-> COLD
```

KUs are placed by rendezvous hashing; with R = 2 each KU is loaded on the first two hosts of its order, with R = 1 its
package (parameters, slots, pointers; never text) is kept as a cold copy on the next host's disk, written in numbered
generations. When probes confirm a death, the survivors take over the dead host's KUs (from a hot copy or by loading
cold copies) through the same MEMBERS phases as live rebalancing, and the gateway re-asks questions that went
incomplete; the tests require identical answers and no lost request. Primary/backup roles with epochs (a backup
promotes itself, epoch + 1) belong to the replication of the text-reading baseline (`mxn/expert/replica.py`); the
neuron system needs no roles, because every copy is the same package and answers bit-identically.

## 5. Spark-style batching

```mermaid
flowchart LR
  QS["Questions"] --> WV["Waves of 250"]
  WV --> PLN["Planner: whole wave"]
  PLN --> RB["One batched RESOLVE broadcast per wave"]
  RB --> TB["One TASKS batch per host"]
  subgraph HOST["Each host"]
    TB --> WK["Worker threads run independent KU tasks in parallel"]
    WK --> FW["Next hops grouped: one TASKS message per target host"]
    WK --> FD["Findings grouped: one FINDINGS message"]
  end
  FW --> TB
  FD --> RES["Resultant"]
  RES --> END["Wave end: retry incomplete, free memory, next wave"]
```

The driver sends questions in waves (default 250): one planner pass, one batched RESOLVE broadcast and one TASKS batch
per host per wave, like a Spark stage. Each host runs its KUs' independent tasks in parallel on worker threads and
groups what they produce, so next hops travel as one message per target host and findings as one message to the
resultant. This batching is why throughput is ~0.07 s per question on 5 host processes while one question alone
waits for its hops.
