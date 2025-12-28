#!/usr/bin/env python
"""
Run all SQNT demonstration scripts and generate figures.

This script:
1. Runs the ground-truth mixture recovery experiments
2. Generates all canonical figures
3. Prints a summary of results

All outputs are deterministic and reproducible (seed=0).
"""

import sys
import argparse
from pathlib import Path

# Ensure package and scripts are importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def run_recovery_experiments(constraint_cfg=None):
    """Run the mixture recovery experiments and generate figures."""
    print("\n" + "=" * 60)
    print("SQNT Ground-Truth Mixture Recovery")
    print("=" * 60)

    # Print constraint status if enabled
    if constraint_cfg is not None and constraint_cfg.get("enabled", False):
        print(f"\nConstraints: enabled={constraint_cfg['enabled']}, "
              f"max_degree={constraint_cfg.get('max_degree', None)}, "
              f"locality_radius={constraint_cfg.get('locality_radius', None)}, "
              f"lambda={constraint_cfg.get('lambda_constraint', 0.0)}")

    import numpy as np
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    # Configuration
    SEED = 0
    N_NODES = 12
    BATCH_SIZE = 1024
    EPOCHS = 300
    TOPOLOGIES = ["chain", "ring", "star", "complete"]

    print(f"\nConfiguration: n={N_NODES}, batch={BATCH_SIZE}, epochs={EPOCHS}")
    print(f"Topologies: {TOPOLOGIES}\n")

    # Sample ground truth
    K = len(TOPOLOGIES)
    w_true = sample_ground_truth_mixture(K, seed=SEED, concentration=0.3)
    print("Ground-truth mixture weights:")
    for k, topo in enumerate(TOPOLOGIES):
        print(f"  {topo:10}: {w_true[k]:.4f}")

    # Generate data
    print("\nGenerating planted mixture data...")
    X, y, _, _ = generate_planted_mixture_data(
        n=N_NODES,
        batch=BATCH_SIZE,
        w_true=w_true,
        topology_names=TOPOLOGIES,
        seed=SEED,
        noise_level=0.05,
    )
    print(f"  Data shape: X={X.shape}, y={y.shape}")
    print(f"  Label balance: {y.mean():.2%} positive")

    # Train to recover
    print("\nTraining to recover mixture...")
    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=TOPOLOGIES,
        n=N_NODES,
        epochs=EPOCHS,
        lr_params=0.2,
        lr_mixture=0.15,
        seed=SEED,
        constraint_cfg=constraint_cfg,
    )

    # Results
    final_weights = history['weights'][-1]
    final_l1 = history['recovery_l1'][-1]
    final_acc = history['acc'][-1]

    print("\n" + "-" * 40)
    print("RECOVERY RESULTS")
    print("-" * 40)
    print("\nLearned mixture weights:")
    for k, topo in enumerate(TOPOLOGIES):
        diff = final_weights[k] - w_true[k]
        print(f"  {topo:10}: {final_weights[k]:.4f} (true: {w_true[k]:.4f}, diff: {diff:+.4f})")

    print(f"\nFinal metrics:")
    print(f"  L1 recovery error: {final_l1:.4f}")
    print(f"  Training accuracy: {final_acc:.4f}")

    return {
        'w_true': w_true,
        'w_learned': final_weights,
        'recovery_l1': final_l1,
        'accuracy': final_acc,
        'history': history,
    }


def run_figure_generation():
    """Generate all canonical figures."""
    print("\n" + "=" * 60)
    print("Generating Canonical Figures")
    print("=" * 60)

    # Import and run the figure script
    from make_figures_recovery import (
        make_recovery_convergence_figure,
        make_phase_diagram_figure,
        make_learned_graph_figure,
        make_topology_confusion_baselines_figure,
        FIGURES_DIR,
        RESULTS_DIR,
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print()
    make_recovery_convergence_figure()
    print()
    make_phase_diagram_figure()
    print()
    make_learned_graph_figure()
    print()
    make_topology_confusion_baselines_figure()


def main():
    parser = argparse.ArgumentParser(
        description="Run SQNT mixture recovery experiments with optional hardware constraints."
    )
    parser.add_argument(
        "--enable-constraints",
        action="store_true",
        default=False,
        help="Enable hardware topology constraints",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=None,
        help="Maximum node degree constraint (e.g., 3)",
    )
    parser.add_argument(
        "--locality-radius",
        type=int,
        default=None,
        help="Maximum hop distance for edges (e.g., 2)",
    )
    parser.add_argument(
        "--lambda-constraint",
        type=float,
        default=0.0,
        help="Constraint penalty strength (default: 0.0)",
    )

    args = parser.parse_args()

    # Validate constraint configuration
    if args.enable_constraints:
        if args.max_degree is None and args.locality_radius is None:
            print("ERROR: --enable-constraints requires at least one of --max-degree or --locality-radius")
            sys.exit(1)

    # Build constraint config
    constraint_cfg = None
    if args.enable_constraints:
        constraint_cfg = {
            "enabled": True,
            "max_degree": args.max_degree,
            "locality_radius": args.locality_radius,
            "lambda_constraint": args.lambda_constraint,
        }

    print("=" * 60)
    print("SQNT Hardware Demo - Full Run")
    print("=" * 60)
    print("\nThis script runs all experiments and generates figures.")
    print("All outputs are deterministic (seed=0).")

    # Run experiments
    results = run_recovery_experiments(constraint_cfg=constraint_cfg)

    # Generate figures
    run_figure_generation()

    # Summary
    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)

    print("\nGenerated figures:")
    print("  figures/sqnt_mixture_recovery_convergence.png")
    print("  figures/sqnt_recovery_phase_diagram.png")
    print("  figures/sqnt_learned_graph_overlay.png")
    print("  figures/sqnt_topology_confusion_baselines.png")

    print("\nKey result: Behavioral learning succeeds; mixture identifiability is measurable.")
    print(f"  Final L1 error: {results['recovery_l1']:.4f}")

    print("\nTo reproduce:")
    print("  pip install -e .")
    print("  python scripts/run_all.py")

    return results


if __name__ == "__main__":
    main()
