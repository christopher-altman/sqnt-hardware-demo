#!/usr/bin/env python
"""
Generate canonical figures for SQNT mixture recovery experiments.

Produces:
1. figures/sqnt_mixture_recovery_convergence.png - Weight convergence plot
2. figures/sqnt_recovery_phase_diagram.png - Identifiability heatmap
3. figures/sqnt_learned_graph_overlay.png - Learned graph visualization
4. figures/sqnt_topology_confusion_baselines.png - Single-topology baselines

All outputs are deterministic (seed=0).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# Ensure package is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqnt_hardware_demo.experiments import (
    sample_ground_truth_mixture,
    generate_planted_mixture_data,
    train_mixture_recovery,
    run_recovery_phase_diagram,
)
from sqnt_hardware_demo.graphs import make_graph_mask
from sqnt_hardware_demo.identifiability import (
    compute_support_metrics,
    run_topology_baselines,
    format_support_report,
    save_identifiability_metrics,
)


# Configuration
SEED = 0
N_NODES = 12
BATCH_SIZE = 1024
EPOCHS = 300
TOPOLOGIES = ["chain", "ring", "star", "complete"]
FIGURES_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR = Path(__file__).parent.parent / "results"
TAU_SUPPORT = 0.05  # Support threshold


def make_recovery_convergence_figure():
    """
    Create the convergence figure showing mixture weight learning over training.

    Shows w_hat_k(t) approaching w_true_k for each topology, with honest
    annotations about recovery quality.
    """
    print("Generating Figure 1: Mixture Recovery Convergence...")

    K = len(TOPOLOGIES)

    # Sample ground truth mixture (non-uniform for visibility)
    w_true = sample_ground_truth_mixture(K, seed=SEED, concentration=0.3)
    print(f"  Ground truth: {dict(zip(TOPOLOGIES, w_true.round(3)))}")

    # Generate planted mixture data
    X, y, _, _ = generate_planted_mixture_data(
        n=N_NODES,
        batch=BATCH_SIZE,
        w_true=w_true,
        topology_names=TOPOLOGIES,
        seed=SEED,
        noise_level=0.05,
    )

    # Train to recover mixture
    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=TOPOLOGIES,
        n=N_NODES,
        epochs=EPOCHS,
        lr_params=0.2,
        lr_mixture=0.15,
        seed=SEED,
    )

    # Compute support metrics
    final_weights = history['weights'][-1]
    final_l1 = history['recovery_l1'][-1]
    final_kl = history['recovery_kl'][-1]
    support_metrics = compute_support_metrics(w_true, final_weights, TAU_SUPPORT)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs_x = np.arange(1, EPOCHS + 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Left plot: weight trajectories
    for k, (topo, color) in enumerate(zip(TOPOLOGIES, colors)):
        ax1.plot(epochs_x, history['weights'][:, k], color=color, linewidth=2,
                 label=f'{topo} (learned)')
        ax1.axhline(w_true[k], color=color, linestyle='--', alpha=0.7,
                    label=f'{topo} (true)')

    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('Mixture Weight $w_k$', fontsize=12)
    ax1.set_title('Learned Weights Approach Ground Truth', fontsize=14)
    ax1.set_xlim(1, EPOCHS)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='center right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Right plot: recovery error
    ax2.semilogy(epochs_x, history['recovery_l1'], 'b-', linewidth=2, label='L1 Error')
    ax2.semilogy(epochs_x, history['recovery_kl'], 'r--', linewidth=2, label='KL Divergence')
    ax2.set_xlabel('Training Epoch', fontsize=12)
    ax2.set_ylabel('Recovery Error', fontsize=12)
    ax2.set_title('Recovery Error Decreases Over Training', fontsize=14)
    ax2.set_xlim(1, EPOCHS)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    # Add metrics annotation
    metrics_text = (
        f"Final L1={final_l1:.3f}, KL={final_kl:.3f}\n"
        f"Support (tau={TAU_SUPPORT}): P={support_metrics['precision']:.2f}, "
        f"R={support_metrics['recall']:.2f}, F1={support_metrics['f1']:.2f}"
    )
    ax2.text(0.98, 0.02, metrics_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    out_path = FIGURES_DIR / "sqnt_mixture_recovery_convergence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")

    # Print final recovery with support metrics
    print(f"  Final weights: {dict(zip(TOPOLOGIES, final_weights.round(3)))}")
    print(f"  Final L1 error: {final_l1:.4f}")
    print(f"  Support precision: {support_metrics['precision']:.3f}")
    print(f"  Support recall: {support_metrics['recall']:.3f}")
    print(f"  Support F1: {support_metrics['f1']:.3f}")

    return history, w_true, final_weights


def make_phase_diagram_figure():
    """
    Create phase diagram showing recovery success vs noise and dataset size.
    """
    print("Generating Figure 2: Recovery Phase Diagram...")

    # Use coarser grid for speed
    noise_levels = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    dataset_sizes = np.array([64, 128, 256, 512, 1024])

    results = run_recovery_phase_diagram(
        n=N_NODES,
        epochs=150,  # Fewer epochs for speed
        topology_names=TOPOLOGIES,
        noise_levels=noise_levels,
        dataset_sizes=dataset_sizes,
        seed=SEED,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Recovery error heatmap
    im1 = ax1.imshow(results['recovery_l1_grid'], aspect='auto', origin='lower',
                     cmap='RdYlGn_r', vmin=0, vmax=1.5)
    ax1.set_xticks(range(len(dataset_sizes)))
    ax1.set_xticklabels(dataset_sizes.astype(int))
    ax1.set_yticks(range(len(noise_levels)))
    ax1.set_yticklabels([f'{n:.0%}' for n in noise_levels])
    ax1.set_xlabel('Dataset Size', fontsize=12)
    ax1.set_ylabel('Label Noise', fontsize=12)
    ax1.set_title('Recovery Error (L1)', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='L1 Distance')

    # Add "recoverable" contour if there are values below threshold
    if np.any(results['recovery_l1_grid'] < 0.3):
        try:
            contour = ax1.contour(results['recovery_l1_grid'], levels=[0.3],
                                  colors='white', linewidths=2)
            ax1.clabel(contour, fmt='Recoverable', fontsize=10)
        except ValueError:
            pass  # No contour if values don't cross threshold

    # Right: Accuracy heatmap
    im2 = ax2.imshow(results['accuracy_grid'], aspect='auto', origin='lower',
                     cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax2.set_xticks(range(len(dataset_sizes)))
    ax2.set_xticklabels(dataset_sizes.astype(int))
    ax2.set_yticks(range(len(noise_levels)))
    ax2.set_yticklabels([f'{n:.0%}' for n in noise_levels])
    ax2.set_xlabel('Dataset Size', fontsize=12)
    ax2.set_ylabel('Label Noise', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='Accuracy')

    fig.suptitle('SQNT Recovery Depends on Data Quality and Quantity', fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = FIGURES_DIR / "sqnt_recovery_phase_diagram.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wrote {out_path}")


def make_learned_graph_figure():
    """
    Create visualization of learned graph structure.

    Shows nodes on a circle with edges weighted by learned mixture mask.
    Includes honest annotation of support metrics and any spurious components.
    """
    print("Generating Figure 3: Learned Graph Overlay...")

    K = len(TOPOLOGIES)

    # Sample ground truth and train
    w_true = sample_ground_truth_mixture(K, seed=SEED, concentration=0.3)

    X, y, _, _ = generate_planted_mixture_data(
        n=N_NODES,
        batch=BATCH_SIZE,
        w_true=w_true,
        topology_names=TOPOLOGIES,
        seed=SEED,
        noise_level=0.05,
    )

    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=TOPOLOGIES,
        n=N_NODES,
        epochs=200,
        seed=SEED,
    )

    # Build masks and compute learned/true mixtures
    masks = [make_graph_mask(name, N_NODES, include_self=True, normalize=True)
             for name in TOPOLOGIES]

    w_learned = history['weights'][-1]
    l1_error = history['recovery_l1'][-1]

    M_true = sum(w_true[k] * masks[k] for k in range(K))
    M_learned = sum(w_learned[k] * masks[k] for k in range(K))

    # Compute support metrics
    support_metrics = compute_support_metrics(w_true, w_learned, TAU_SUPPORT)

    # Identify spurious components (in learned support but not true support)
    spurious = support_metrics['support_learned'] - support_metrics['support_true']
    spurious_text = ""
    if spurious:
        spurious_topos = [TOPOLOGIES[k] for k in spurious]
        spurious_weights = [w_learned[k] for k in spurious]
        spurious_text = f"\nSpurious: {', '.join(f'{t}={w:.3f}' for t, w in zip(spurious_topos, spurious_weights))}"

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, M, title in [(axes[0], M_true, 'Ground Truth Mixture'),
                          (axes[1], M_learned, 'Learned Mixture')]:
        # Position nodes on a circle
        angles = np.linspace(0, 2 * np.pi, N_NODES, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)

        # Draw edges with weight-based opacity
        max_weight = M.max()
        for i in range(N_NODES):
            for j in range(i + 1, N_NODES):
                weight = (M[i, j] + M[j, i]) / 2  # Symmetrize
                if weight > 0.01:
                    alpha = min(1.0, weight / max_weight)
                    linewidth = 1 + 3 * (weight / max_weight)
                    ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                            'b-', alpha=alpha * 0.6, linewidth=linewidth)

        # Draw nodes
        ax.scatter(x_pos, y_pos, s=300, c='white', edgecolors='black',
                   linewidth=2, zorder=5)

        # Label nodes
        for i in range(N_NODES):
            ax.annotate(str(i), (x_pos[i], y_pos[i]),
                        ha='center', va='center', fontsize=10, zorder=6)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    # Add metrics box with honest annotations
    metrics_box = (
        f"L1 error: {l1_error:.3f}\n"
        f"Support P/R/F1: {support_metrics['precision']:.2f}/{support_metrics['recall']:.2f}/{support_metrics['f1']:.2f}"
        f"{spurious_text}"
    )

    # Add weight comparison
    weight_text = "Weights:\n"
    for k, topo in enumerate(TOPOLOGIES):
        in_true = "+" if k in support_metrics['support_true'] else "-"
        in_learned = "+" if k in support_metrics['support_learned'] else "-"
        weight_text += f"  {topo}: true={w_true[k]:.3f}, learned={w_learned[k]:.3f} [{in_true}/{in_learned}]\n"

    fig.text(0.5, 0.02, weight_text, ha='center', fontsize=9,
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Determine honest title based on recovery quality
    if l1_error < 0.1:
        main_title = 'Learned Topology Closely Matches Ground Truth'
    elif l1_error < 0.3:
        main_title = 'Learned Topology Approximates Ground Truth'
    else:
        main_title = 'Learned Topology Differs from Ground Truth'

    fig.suptitle(main_title, fontsize=16, y=0.98)

    # Add smaller subtitle with metrics
    fig.text(0.5, 0.93, metrics_box, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    fig.tight_layout(rect=[0, 0.12, 1, 0.90])
    out_path = FIGURES_DIR / "sqnt_learned_graph_overlay.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wrote {out_path}")

    return w_true, w_learned, l1_error, support_metrics


def make_topology_confusion_baselines_figure():
    """
    Create a figure showing single-topology baseline accuracies.

    This diagnostic reveals topology confusability: if multiple topologies
    achieve similar accuracy, they may be substitutable under the observable.
    """
    print("Generating Figure 4: Topology Confusion Baselines...")

    K = len(TOPOLOGIES)

    # Sample ground truth
    w_true = sample_ground_truth_mixture(K, seed=SEED, concentration=0.3)

    # Generate planted mixture data
    X, y, _, _ = generate_planted_mixture_data(
        n=N_NODES,
        batch=BATCH_SIZE,
        w_true=w_true,
        topology_names=TOPOLOGIES,
        seed=SEED,
        noise_level=0.05,
    )

    # Run baselines
    baseline_results = run_topology_baselines(
        X, y,
        topology_names=TOPOLOGIES,
        n=N_NODES,
        epochs=150,
        lr=0.2,
        seed=SEED,
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    topos = list(baseline_results.keys())
    accs = [baseline_results[t]['acc'] for t in topos]
    best_accs = [baseline_results[t]['best_acc'] for t in topos]

    x = np.arange(len(topos))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label='Final Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, best_accs, width, label='Best Accuracy', color='lightsteelblue')

    ax.set_xlabel('Topology', fontsize=12)
    ax.set_ylabel('Training Accuracy', fontsize=12)
    ax.set_title('Single-Topology Baseline Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(topos)
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Add annotation about confusability
    acc_range = max(accs) - min(accs)
    if acc_range < 0.05:
        confusion_note = "High confusability: all topologies achieve similar accuracy"
    elif acc_range < 0.15:
        confusion_note = "Moderate confusability: some topologies are substitutable"
    else:
        confusion_note = "Low confusability: topologies are distinguishable"

    ax.text(0.5, 0.02, confusion_note, transform=ax.transAxes, fontsize=10,
            ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Add ground truth annotation
    gt_text = "Ground truth: " + ", ".join(f"{t}={w_true[k]:.2f}" for k, t in enumerate(topos))
    ax.text(0.5, 0.97, gt_text, transform=ax.transAxes, fontsize=9,
            ha='center', va='top')

    fig.tight_layout()
    out_path = FIGURES_DIR / "sqnt_topology_confusion_baselines.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")

    # Print results
    print("  Baseline accuracies:")
    for topo in topos:
        print(f"    {topo}: final={baseline_results[topo]['acc']:.4f}, "
              f"best={baseline_results[topo]['best_acc']:.4f}")

    return baseline_results


def main():
    print("=" * 60)
    print("SQNT Mixture Recovery Figures")
    print("=" * 60)
    print(f"Configuration: n={N_NODES}, epochs={EPOCHS}, seed={SEED}")
    print(f"Topologies: {TOPOLOGIES}")
    print(f"Support threshold (tau): {TAU_SUPPORT}")
    print()

    # Ensure directories exist
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Generate figures
    history, w_true, w_learned = make_recovery_convergence_figure()
    print()
    make_phase_diagram_figure()
    print()
    w_true_graph, w_learned_graph, l1_error, support_metrics = make_learned_graph_figure()
    print()
    baseline_results = make_topology_confusion_baselines_figure()

    # Save identifiability metrics to JSON
    print()
    print("Saving identifiability metrics...")
    save_identifiability_metrics(
        str(RESULTS_DIR / "identifiability_metrics.json"),
        w_true=w_true_graph,
        w_learned=w_learned_graph,
        topology_names=TOPOLOGIES,
        l1_error=l1_error,
        kl_div=history['recovery_kl'][-1],
        accuracy=history['acc'][-1],
        tau=TAU_SUPPORT,
        baseline_results=baseline_results,
    )
    print(f"  Wrote {RESULTS_DIR / 'identifiability_metrics.json'}")

    # Print support report
    print()
    print(format_support_report(w_true_graph, w_learned_graph, TOPOLOGIES, TAU_SUPPORT))

    print()
    print("=" * 60)
    print("All figures generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
