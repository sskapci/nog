"""KUs as neurons (design reports/design_ku_neurons.md, approved with rulings N1-N3 on 2026-10-30).

Each KU owns its answer slots (built from its own article only), its training examples (arm S: cloze variants of its
own sentences) and a small MLP mapping a query signal (shared encoder vector + relation code) to one of its slots or
"none". Nothing here reads another KU.
"""
