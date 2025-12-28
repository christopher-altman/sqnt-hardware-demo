# sqnt-hardware-demo

A minimal, runnable *bridge artifact* connecting the theory line from:

- **Superpositional Quantum Network Topologies** (IJTP 2004)
- **Backpropagation in Adaptive Quantum Networks** (IJTP 2010)
- **Accelerated Training Convergence in Superposed Quantum Networks** (NATO ASI)

to a modern, reviewable demonstration.

## What this repo demonstrates (v1)

1. Construct a small **family of graph topologies** (chain, ring, star, complete).
2. Define an **operator-space weight matrix** \(W\) and “spatialize” it onto each graph via a topology mask.
3. Form a **superposed topology** by mixing masks with a single mixture parameter \(\alpha\).
4. Train a tiny model on a synthetic task and output a single canonical figure:

**`figures/sqnt_mixture_curve.png`**: accuracy vs topology mixture parameter.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/make_figures.py
```
