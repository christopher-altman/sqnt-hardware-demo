#!/usr/bin/env python
"""
Run all SQNT demonstration scripts and generate figures.

This script:
1. Runs the original alpha-sweep figure generation
2. Runs the learned mixture demonstration
3. Prints a summary of results

All outputs are deterministic and reproducible.
"""

import sys
from pathlib import Path

# Ensure we can import from scripts directory
SCRIPTS_DIR = Path(__file__).parent


def run_original_figure():
    """Run the original make_figures.py script."""
    print("\n" + "=" * 60)
    print("PART 1: Original Alpha-Sweep Figure")
    print("=" * 60)

    from sqnt_hardware_demo.train_demo import sweep_alphas
    import numpy as np
    import matplotlib.pyplot as plt

    alphas = np.linspace(0.0, 1.0, 11)
    accs = sweep_alphas(alphas, n=12, seed=0, topo0="chain", topo1="complete")

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(alphas, accs, marker="o")
    plt.xlabel("Topology mixture parameter alpha  (chain -> complete)")
    plt.ylabel("Training-set accuracy")
    plt.title("SQNT demo: accuracy vs superposed topology")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    out = figures_dir / "sqnt_mixture_curve.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")

    print("\nAlpha-sweep results:")
    for a, acc in zip(alphas, accs):
        print(f"  alpha={a:.1f}: acc={acc:.4f}")


def run_learned_mixture():
    """Run the learned mixture demonstration."""
    print("\n" + "=" * 60)
    print("PART 2: Learned Mixture Demonstration")
    print("=" * 60)

    # Import and run the learned mixture script
    import numpy as np
    import matplotlib.pyplot as plt
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

    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    print(f"\nConfiguration: n={N_NODES}, epochs={EPOCHS}, seed={SEED}")
    print(f"Topologies: {TOPOLOGIES}\n")

    # Generate data
    X, y = make_synthetic(n=N_NODES, batch=BATCH_SIZE, seed=SEED)

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

    # Train baselines
    print("Training fixed topology baselines...")
    fixed_results = {}
    for topo in TOPOLOGIES:
        acc, hist = train_fixed_topology(
            X, y, topology_name=topo, n=N_NODES, epochs=EPOCHS, lr=LR_PARAMS, seed=SEED
        )
        fixed_results[topo] = (acc, hist)

    best_fixed_name = max(fixed_results, key=lambda k: fixed_results[k][0])
    best_fixed_acc, best_fixed_hist = fixed_results[best_fixed_name]

    # Train random mixture
    print("Training random mixture control...")
    random_acc, random_hist, frozen_weights = train_random_mixture(
        X, y, topology_names=TOPOLOGIES, n=N_NODES, epochs=EPOCHS, lr=LR_PARAMS, seed=SEED
    )

    # Generate figures
    print("\nGenerating figures...")
    epochs_x = np.arange(1, EPOCHS + 1)

    # Figure 1: Performance comparison
    fig1, ax1 = plt.subplots(figsize=(8, 5))
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
    out1 = figures_dir / "sqnt_learned_mixture_curve.png"
    fig1.savefig(out1, dpi=200)
    plt.close(fig1)
    print(f"  Wrote {out1}")

    # Figure 2: Mixture weights
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    weights_array = np.array(learned_hist['weights'])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax2.stackplot(epochs_x, weights_array.T, labels=TOPOLOGIES, colors=colors, alpha=0.7)
    ax2.set_xlabel('Training Epoch', fontsize=12)
    ax2.set_ylabel('Mixture Weight', fontsize=12)
    ax2.set_title('SQNT: Learned Topology Mixture Weights Over Training', fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(1, EPOCHS)
    ax2.legend(loc='center right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    fig2.tight_layout()
    out2 = figures_dir / "sqnt_mixture_weights.png"
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)
    print(f"  Wrote {out2}")

    # Print summary
    final_weights = learned_hist['weights'][-1]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nFinal Accuracies:")
    print(f"  Learned mixture:      {learned_acc:.4f}")
    print(f"  Best fixed ({best_fixed_name:8}): {best_fixed_acc:.4f}")
    print(f"  Random mixture:       {random_acc:.4f}")
    print(f"\nFixed topology breakdown:")
    for topo in TOPOLOGIES:
        print(f"  {topo:10}: {fixed_results[topo][0]:.4f}")
    print(f"\nFinal learned mixture weights:")
    for i, topo in enumerate(TOPOLOGIES):
        print(f"  {topo:10}: {final_weights[i]:.4f}")
    print(f"\nFrozen random weights:")
    for i, topo in enumerate(TOPOLOGIES):
        print(f"  {topo:10}: {frozen_weights[i]:.4f}")

    return {
        'learned_acc': learned_acc,
        'best_fixed_acc': best_fixed_acc,
        'best_fixed_name': best_fixed_name,
        'random_acc': random_acc,
        'final_weights': final_weights,
    }


def main():
    print("=" * 60)
    print("SQNT Hardware Demo - Full Run")
    print("=" * 60)
    print("\nThis script generates all figures and prints results.")
    print("All outputs are deterministic (seed=0).")

    run_original_figure()
    results = run_learned_mixture()

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print("\nGenerated figures:")
    print("  figures/sqnt_mixture_curve.png")
    print("  figures/sqnt_learned_mixture_curve.png")
    print("  figures/sqnt_mixture_weights.png")
    print("\nTo reproduce:")
    print("  pip install -e .")
    print("  python scripts/run_all.py")

    return results


if __name__ == "__main__":
    main()
