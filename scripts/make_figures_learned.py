#!/usr/bin/env python
"""
Generate canonical figures for SQNT learned mixture demonstration.

Produces:
1. figures/sqnt_learned_mixture_curve.png - Performance comparison
2. figures/sqnt_mixture_weights.png - Learned weight evolution

All outputs are deterministic (seed=0).
Runtime: <60s on typical hardware.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sqnt_hardware_demo.train_demo import make_synthetic
from sqnt_hardware_demo.mixture import (
    train_learned_mixture,
    train_fixed_topology,
    train_random_mixture,
)


# Configuration
SEED = 0
N_NODES = 12
BATCH_SIZE = 512
EPOCHS = 200
LR_PARAMS = 0.2
LR_MIXTURE = 0.15
TOPOLOGIES = ["chain", "ring", "star", "complete"]

# Ensure figures directory exists
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    print("SQNT Learned Mixture Demo")
    print("=" * 50)
    print(f"Configuration: n={N_NODES}, epochs={EPOCHS}, seed={SEED}")
    print(f"Topologies: {TOPOLOGIES}")
    print()

    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = make_synthetic(n=N_NODES, batch=BATCH_SIZE, seed=SEED)
    print(f"  Data shape: X={X.shape}, y={y.shape}")
    print()

    # Train learned mixture
    print("Training learned mixture...")
    learned_acc, learned_hist = train_learned_mixture(
        X, y,
        topology_names=TOPOLOGIES,
        n=N_NODES,
        epochs=EPOCHS,
        lr_params=LR_PARAMS,
        lr_mixture=LR_MIXTURE,
        seed=SEED,
    )
    print(f"  Final accuracy: {learned_acc:.4f}")
    final_weights = learned_hist['weights'][-1]
    for i, topo in enumerate(TOPOLOGIES):
        print(f"  Weight[{topo}]: {final_weights[i]:.4f}")
    print()

    # Train fixed topology baselines
    print("Training fixed topology baselines...")
    fixed_results = {}
    for topo in TOPOLOGIES:
        acc, hist = train_fixed_topology(
            X, y,
            topology_name=topo,
            n=N_NODES,
            epochs=EPOCHS,
            lr=LR_PARAMS,
            seed=SEED,
        )
        fixed_results[topo] = (acc, hist)
        print(f"  {topo}: {acc:.4f}")

    # Find best fixed topology
    best_fixed_name = max(fixed_results, key=lambda k: fixed_results[k][0])
    best_fixed_acc, best_fixed_hist = fixed_results[best_fixed_name]
    print(f"  Best fixed: {best_fixed_name} ({best_fixed_acc:.4f})")
    print()

    # Train random mixture control
    print("Training random mixture control...")
    random_acc, random_hist, frozen_weights = train_random_mixture(
        X, y,
        topology_names=TOPOLOGIES,
        n=N_NODES,
        epochs=EPOCHS,
        lr=LR_PARAMS,
        seed=SEED,
    )
    print(f"  Final accuracy: {random_acc:.4f}")
    print(f"  Frozen weights: {frozen_weights}")
    print()

    # Create Figure 1: Performance comparison
    print("Generating Figure 1: Performance comparison...")
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    epochs_x = np.arange(1, EPOCHS + 1)

    ax1.plot(epochs_x, learned_hist['acc'], 'b-', linewidth=2,
             label=f'Learned Mixture ({learned_acc:.3f})')
    ax1.plot(epochs_x, best_fixed_hist['acc'], 'g--', linewidth=1.5,
             label=f'Best Fixed ({best_fixed_name}, {best_fixed_acc:.3f})')
    ax1.plot(epochs_x, random_hist['acc'], 'r:', linewidth=1.5,
             label=f'Random Mixture ({random_acc:.3f})')

    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy', fontsize=12)
    ax1.set_title('SQNT: Learned vs Fixed Topology Performance', fontsize=14)
    ax1.set_ylim(0.4, 1.02)
    ax1.set_xlim(1, EPOCHS)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    out1 = FIGURES_DIR / "sqnt_learned_mixture_curve.png"
    fig1.savefig(out1, dpi=200)
    plt.close(fig1)
    print(f"  Wrote {out1}")

    # Create Figure 2: Mixture weight evolution
    print("Generating Figure 2: Mixture weight evolution...")
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    weights_array = np.array(learned_hist['weights'])  # shape (epochs, K)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Stacked area chart
    ax2.stackplot(epochs_x, weights_array.T, labels=TOPOLOGIES,
                  colors=colors, alpha=0.7)

    ax2.set_xlabel('Training Epoch', fontsize=12)
    ax2.set_ylabel('Mixture Weight', fontsize=12)
    ax2.set_title('SQNT: Learned Topology Mixture Weights Over Training', fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(1, EPOCHS)
    ax2.legend(loc='center right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    fig2.tight_layout()
    out2 = FIGURES_DIR / "sqnt_mixture_weights.png"
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)
    print(f"  Wrote {out2}")

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Learned mixture:    {learned_acc:.4f}")
    print(f"Best fixed ({best_fixed_name:8}): {best_fixed_acc:.4f}")
    print(f"Random mixture:     {random_acc:.4f}")
    print()
    print("Final learned weights:")
    for i, topo in enumerate(TOPOLOGIES):
        print(f"  {topo:10}: {final_weights[i]:.4f}")
    print()
    print("Figures written:")
    print(f"  {out1}")
    print(f"  {out2}")


if __name__ == "__main__":
    main()
