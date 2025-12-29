#!/usr/bin/env python
"""
Run all SQNT demonstration scripts and generate figures.

This script:
1. Runs the ground-truth mixture recovery experiments
2. Generates all canonical figures
3. Prints a summary of results

All outputs are deterministic and reproducible (seed=0).

Phase III flags (opt-in, defaults OFF):
  --enable-compile-constraints   Enable compilation-aware constraints
  --device-graph PATH            Path to device graph JSON
  --lambda-compile FLOAT         Compilation penalty strength
  --compile-target {none,qiskit,pennylane,cirq}
  --simulate-with {none,qiskit,pennylane,cirq}
  --shots INT                    Shots for simulation

Phase IV flags (opt-in, defaults OFF):
  --enable-multi-observable      Enable multi-observable training
  --lambda-aux FLOAT             Auxiliary loss weight
  --aux-task STR                 Auxiliary task type
  --aux-seed INT                 Auxiliary random seed

Phase V flags (opt-in, defaults OFF):
  --enable-adaptive-topology     Enable adaptive topology dynamics
  --adaptive-beta FLOAT          Inertia/EMA parameter
  --adaptive-momentum FLOAT      Momentum parameter
  --adaptive-update {momentum,ema}
"""

import argparse
import sys
from pathlib import Path

# Ensure package and scripts are importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def run_recovery_experiments(
    enable_compile_constraints=False,
    device_graph_path=None,
    lambda_compile=0.0,
    enable_multi_observable=False,
    lambda_aux=0.0,
    aux_task="graph_feature",
    aux_seed=0,
    enable_adaptive_topology=False,
    adaptive_beta=0.0,
    adaptive_momentum=0.0,
    adaptive_update="momentum",
):
    """Run the mixture recovery experiments and generate figures."""
    print("\n" + "=" * 60)
    print("SQNT Ground-Truth Mixture Recovery")
    print("=" * 60)

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

    # Phase III: Load device graph if provided
    device_graph = None
    if enable_compile_constraints and device_graph_path:
        from sqnt_hardware_demo.compilation import load_device_graph
        try:
            device_graph = load_device_graph(device_graph_path)
            print(f"Loaded device graph: {device_graph.get('name', 'unknown')}")
            print(f"  Nodes: {len(device_graph.get('nodes', []))}")
            print(f"  Edges: {len(device_graph.get('edges', []))}")
            print(f"  Lambda compile: {lambda_compile}")
        except FileNotFoundError:
            print(f"Warning: Device graph not found at {device_graph_path}")
            print("  Continuing without compilation constraints.")
            device_graph = None

    # Sample ground truth
    K = len(TOPOLOGIES)
    w_true = sample_ground_truth_mixture(K, seed=SEED, concentration=0.3)
    print("\nGround-truth mixture weights:")
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
    if enable_multi_observable:
        print(f"  Multi-observable enabled: lambda_aux={lambda_aux}, aux_task={aux_task}")
    if enable_adaptive_topology:
        print(f"  Adaptive topology enabled: beta={adaptive_beta}, momentum={adaptive_momentum}")

    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=TOPOLOGIES,
        n=N_NODES,
        epochs=EPOCHS,
        lr_params=0.2,
        lr_mixture=0.15,
        seed=SEED,
        enable_compile_constraints=enable_compile_constraints,
        device_graph=device_graph,
        lambda_compile=lambda_compile,
        enable_multi_observable=enable_multi_observable,
        lambda_aux=lambda_aux,
        aux_task=aux_task,
        aux_seed=aux_seed,
        enable_adaptive_topology=enable_adaptive_topology,
        adaptive_beta=adaptive_beta,
        adaptive_momentum=adaptive_momentum,
        adaptive_update=adaptive_update,
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

    if 'loss_compile' in history:
        print(f"  Final compile penalty: {history['loss_compile'][-1]:.4f}")

    if 'loss_aux' in history:
        print(f"  Final aux loss: {history['loss_aux'][-1]:.4f}")
        print(f"  Final aux metric (MSE): {history['acc_aux'][-1]:.4f}")

    if 'adaptive_step_norm' in history:
        print(f"  Mean adaptive step norm: {history['adaptive_step_norm'].mean():.4f}")

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


def run_phase3_demo(compile_target, simulate_with, shots, device_graph_path):
    """Run Phase III compilation and simulation demo."""
    print("\n" + "=" * 60)
    print("Phase III: Compilation & Simulation Demo")
    print("=" * 60)

    import numpy as np
    from sqnt_hardware_demo.graphs import make_graph_mask

    # Create a sample topology mask
    n = 6
    mask = make_graph_mask("ring", n, include_self=True, normalize=True)

    # Compile to target backend
    if compile_target != "none":
        print(f"\nCompiling to {compile_target}...")
        try:
            if compile_target == "qiskit":
                from sqnt_hardware_demo.circuit_targets import to_qiskit_circuit
                circuit = to_qiskit_circuit(mask, seed=0)
                print(f"  Created Qiskit circuit with {circuit.num_qubits} qubits")
            elif compile_target == "pennylane":
                from sqnt_hardware_demo.circuit_targets import to_pennylane_qnode
                qnode = to_pennylane_qnode(mask, shots=shots, seed=0)
                print(f"  Created PennyLane QNode for {n} qubits")
            elif compile_target == "cirq":
                from sqnt_hardware_demo.circuit_targets import to_cirq_circuit
                circuit = to_cirq_circuit(mask, seed=0)
                print(f"  Created Cirq circuit")
        except ImportError as e:
            print(f"  Skipping {compile_target}: {e}")

    # Run simulation
    if simulate_with != "none":
        print(f"\nSimulating with {simulate_with} ({shots} shots)...")
        try:
            if simulate_with == "qiskit":
                from sqnt_hardware_demo.circuit_targets import to_qiskit_circuit
                from sqnt_hardware_demo.sim_backends import simulate_qiskit
                circuit = to_qiskit_circuit(mask, seed=0)
                results = simulate_qiskit(circuit, shots=shots)
                print(f"  Got {len(results['counts'])} unique bitstrings")
            elif simulate_with == "pennylane":
                from sqnt_hardware_demo.circuit_targets import to_pennylane_qnode
                from sqnt_hardware_demo.sim_backends import simulate_pennylane
                qnode = to_pennylane_qnode(mask, shots=shots, seed=0)
                results = simulate_pennylane(qnode, shots=shots)
                print(f"  Got {len(results['counts'])} unique bitstrings")
            elif simulate_with == "cirq":
                from sqnt_hardware_demo.circuit_targets import to_cirq_circuit
                from sqnt_hardware_demo.sim_backends import simulate_cirq
                circuit = to_cirq_circuit(mask, seed=0)
                results = simulate_cirq(circuit, shots=shots)
                print(f"  Got {len(results['counts'])} unique bitstrings")
        except ImportError as e:
            print(f"  Skipping {simulate_with} simulation: {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SQNT hardware demo experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Phase III: Compilation constraints
    parser.add_argument(
        "--enable-compile-constraints",
        action="store_true",
        help="Enable compilation-aware constraints in mixture recovery",
    )
    parser.add_argument(
        "--device-graph",
        type=str,
        default=None,
        help="Path to device graph JSON file",
    )
    parser.add_argument(
        "--lambda-compile",
        type=float,
        default=0.0,
        help="Compilation penalty strength (default: 0.0)",
    )

    # Phase III: Compilation targets
    parser.add_argument(
        "--compile-target",
        choices=["none", "qiskit", "pennylane", "cirq"],
        default="none",
        help="Quantum backend for circuit compilation (default: none)",
    )
    parser.add_argument(
        "--simulate-with",
        choices=["none", "qiskit", "pennylane", "cirq"],
        default="none",
        help="Quantum backend for simulation (default: none)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of shots for simulation (default: 1024)",
    )

    # Phase IV: Multi-observable identifiability
    parser.add_argument(
        "--enable-multi-observable",
        action="store_true",
        help="Enable multi-observable training for improved identifiability",
    )
    parser.add_argument(
        "--lambda-aux",
        type=float,
        default=0.0,
        help="Auxiliary loss weight (default: 0.0)",
    )
    parser.add_argument(
        "--aux-task",
        type=str,
        default="graph_feature",
        help="Auxiliary task type (default: graph_feature)",
    )
    parser.add_argument(
        "--aux-seed",
        type=int,
        default=0,
        help="Auxiliary random seed (default: 0)",
    )

    # Phase V: Adaptive topology learning
    parser.add_argument(
        "--enable-adaptive-topology",
        action="store_true",
        help="Enable adaptive topology dynamics (AQN)",
    )
    parser.add_argument(
        "--adaptive-beta",
        type=float,
        default=0.0,
        help="Inertia/EMA parameter for adaptive updates (default: 0.0)",
    )
    parser.add_argument(
        "--adaptive-momentum",
        type=float,
        default=0.0,
        help="Momentum parameter for adaptive updates (default: 0.0)",
    )
    parser.add_argument(
        "--adaptive-update",
        type=str,
        default="momentum",
        choices=["momentum", "ema"],
        help="Adaptive update rule (default: momentum)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SQNT Hardware Demo - Full Run")
    print("=" * 60)
    print("\nThis script runs all experiments and generates figures.")
    print("All outputs are deterministic (seed=0).")

    # Run experiments
    results = run_recovery_experiments(
        enable_compile_constraints=args.enable_compile_constraints,
        device_graph_path=args.device_graph,
        lambda_compile=args.lambda_compile,
        enable_multi_observable=args.enable_multi_observable,
        lambda_aux=args.lambda_aux,
        aux_task=args.aux_task,
        aux_seed=args.aux_seed,
        enable_adaptive_topology=args.enable_adaptive_topology,
        adaptive_beta=args.adaptive_beta,
        adaptive_momentum=args.adaptive_momentum,
        adaptive_update=args.adaptive_update,
    )

    # Generate figures
    run_figure_generation()

    # Phase III: Optional compilation and simulation demo
    if args.compile_target != "none" or args.simulate_with != "none":
        run_phase3_demo(
            args.compile_target,
            args.simulate_with,
            args.shots,
            args.device_graph,
        )

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

    if args.enable_compile_constraints:
        print("\nPhase III with compilation constraints:")
        print("  python scripts/run_all.py --enable-compile-constraints \\")
        print("      --device-graph device_graphs/line_n12.json --lambda-compile 0.1")

    return results


if __name__ == "__main__":
    main()
