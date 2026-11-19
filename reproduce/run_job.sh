#!/bin/bash
# Re-run one headline job exactly as it ran: the source archive of the code at the job's commit (reproduce/jobs.tsv;
# archives attached to this snapshot's GitHub release, checksums in reproduce/SHA256SUMS), unpacked into WORK_DIR/JOB,
# and the original job script (reproduce/original/<job>.sh) with its working-directory line pointed there.
# usage: reproduce/run_job.sh JOB DATA_DIR VENV_DIR [WORK_DIR]   (REPRODUCE_DRY=1: set up only, do not run)
#   DATA_DIR: the prepared inputs, laid out as the job expects under artifacts/ (see reproduce/README.md "Inputs");
#   VENV_DIR: a Python environment with the packages of pyproject.toml (+ faiss-cpu, sentence-transformers for RAG).
#   REPRODUCE_ARCHIVES=DIR: use archives already downloaded into DIR instead of downloading them.
# Jobs that read another job's outputs (dev → test, gates) need that job run first, into the same DATA_DIR.
set -e
RELEASE=snapshot-2026-11-19
URL=https://github.com/sskapci/nog/releases/download/$RELEASE
JOB=$1; DATA=$(readlink -f "$2"); VENV=$(readlink -f "$3"); WORK=$(readlink -f "${4:-reproduce_work}")
REPO=$(cd "$(dirname "$0")/.." && pwd)
line=$(grep -P "^$JOB\t" "$REPO/reproduce/jobs.tsv") || { echo "unknown job $JOB"; exit 2; }
COMMIT=$(echo "$line" | cut -f2)
ARC=mxn-src-$COMMIT.tar.gz
WT=$WORK/$JOB; mkdir -p "$WORK"
if [ ! -d "$WT" ]; then
    SRC=${REPRODUCE_ARCHIVES:-$WORK/archives}; mkdir -p "$SRC"
    [ -s "$SRC/$ARC" ] || curl -fsSL -o "$SRC/$ARC" "$URL/$ARC" \
        || gh release download "$RELEASE" -R sskapci/nog -p "$ARC" -D "$SRC" \
        || { echo "cannot download $ARC (release $RELEASE)"; exit 3; }
    (cd "$SRC" && grep " $ARC\$" "$REPO/reproduce/SHA256SUMS" | sha256sum -c -) || { echo "checksum mismatch: $ARC"; exit 4; }
    mkdir -p "$WT"; tar -xzf "$SRC/$ARC" -C "$WT" --strip-components=1
fi
ln -sfn "$DATA" "$WT/artifacts"; ln -sfn "$VENV" "$WT/.venv"
# MetaQA RAG question files are not redistributed (ids only): rebuild them from the public copy
if grep -q "reports/rag/mqs_" "$REPO/reproduce/original/$JOB.sh"; then
    "$VENV/bin/python" "$REPO/experiments/baselines/rag_mq_text.py" "$DATA/data/metaqa_sets" "$WT/reports/rag"
fi
sed -E "s#cd /home/[^/]+/projects/t24[a-z0-9-]*#cd $WT#" "$REPO/reproduce/original/$JOB.sh" > "$WORK/$JOB.sh"
echo "job $JOB at $COMMIT ($ARC) in $WT"
[ -n "$REPRODUCE_DRY" ] && { echo "dry: would run $WORK/$JOB.sh"; exit 0; }
bash "$WORK/$JOB.sh"
