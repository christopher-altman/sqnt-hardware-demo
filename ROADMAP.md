# SQNT Roadmap

This document outlines the staged expansion of the SQNT Hardware Demo from a
controlled topology-identifiability testbed into a hardware-aware, compilation-
targeted, and simulation-integrated research platform.

The guiding principle is **methodological separation**:
we first expose identifiability limits under fixed structure, then introduce
mechanisms to resolve those limits in a controlled and falsifiable way.

---

## Phase 0 — Completed (Baseline Identifiability)

**Status:** Complete

- Canonical topologies (chain, ring, star, complete)
- Planted mixture recovery with deterministic runs
- Identifiability diagnostics:
  - L1 recovery error
  - Thresholded support recovery
  - Single-topology confusion baselines
- Honest reporting of accuracy vs. identifiability mismatch

**Key result:**  
High behavioral accuracy can coexist with topology confusability; static recovery
is sometimes ill-posed under limited observables.

---

## Phase I — Hardware-Aware Topology Constraints (Next)

**Goal:**  
Introduce *constraint operators* that restrict the admissible topology space
based on physical or architectural considerations, without introducing adaptivity.

**Scope (minimal, review-proof):**
- Topology feasibility masks (e.g., degree bounds, locality radius)
- Constraint penalties applied to mixture weights
- No adaptive topology learning
- No new observables

**Scientific question:**  
Does restricting the hypothesis space restore identifiability without increasing
model expressivity?

---

## Phase II — Compilation-Targeted Topologies

**Goal:**  
Align SQNT topologies with realizable quantum circuit layouts.

**Examples:**
- Linear nearest-neighbor chains
- Heavy-hex / grid-like interaction graphs
- Limited-degree interaction graphs

**Additions:**
- Mapping from abstract topology → compilation target
- Identifiability measured under compilation constraints

---

## Phase III — Multi-Observable Identifiability

**Goal:**  
Break topology confusability by increasing observability, not model capacity.

**Mechanisms:**
- Auxiliary graph-induced observables (spectral, path-length proxies)
- Multi-task losses sharing topology but differing observables
- Identifiability phase diagrams over observable sets

This phase explicitly treats topology confusability as a *gauge freedom* broken
by additional measurements.

---

## Phase IV — Adaptive Quantum Networks (AQN)

**Goal:**  
Reintroduce topology adaptation as a *response* to demonstrated degeneracy.

**Key distinction:**
- Adaptivity is not assumed
- It is introduced only when static recovery is provably ill-posed

**Connection to prior work:**
- Topology backpropagation
- Adaptive quantum network learning dynamics
- Degeneracy-breaking through co-evolving structure

---

## Long-Horizon Extensions

- Hardware-in-the-loop experiments
- Noise-aware identifiability
- Active experiment design for topology inference
- Closed-loop compilation ↔ learning ↔ constraint feedback