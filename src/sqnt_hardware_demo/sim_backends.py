"""
Quantum simulation backends for SQNT circuit execution.

Provides unified interfaces for simulating circuits with Qiskit,
PennyLane, and Cirq backends. Includes optional noise model support.
"""

import json
import numpy as np
from typing import Dict, Optional, Any


def load_noise_model(path: str) -> dict:
    """
    Load a noise model specification from JSON.

    Expected schema:
    {
        "type": "depolarizing" | "dephasing" | "none",
        "parameters": { ... type-specific ... }
    }

    Parameters
    ----------
    path : str
        Path to noise model JSON file.

    Returns
    -------
    noise_model : dict
        Parsed noise model specification.
    """
    with open(path, 'r') as f:
        return json.load(f)


def simulate_qiskit(
    circuit: Any,
    *,
    shots: int = 1024,
    noise_model: Optional[dict] = None,
) -> dict:
    """
    Simulate a Qiskit circuit and return measurement results.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        The circuit to simulate.
    shots : int
        Number of measurement shots.
    noise_model : dict, optional
        Noise model specification. Supported types:
        - {"type": "none"}: ideal simulation
        - {"type": "depolarizing", "parameters": {"p_error": float}}
        Currently only 'none' is fully supported; other types print
        a warning and fall back to ideal simulation.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'counts': measurement counts as {bitstring: count}
        - 'shots': number of shots
        - 'backend': 'qiskit'
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        try:
            from qiskit.providers.aer import AerSimulator
        except ImportError:
            raise ImportError(
                "Qiskit Aer is not installed. Install with: pip install qiskit-aer"
            )

    backend = AerSimulator()

    # Handle noise model
    qiskit_noise = None
    if noise_model is not None and noise_model.get('type') != 'none':
        noise_type = noise_model.get('type', 'unknown')
        print(f"[sim_backends] Warning: Noise model '{noise_type}' not fully "
              f"implemented for Qiskit. Using ideal simulation.")

    job = backend.run(circuit, shots=shots, noise_model=qiskit_noise)
    result = job.result()
    counts = result.get_counts()

    return {
        'counts': dict(counts),
        'shots': shots,
        'backend': 'qiskit',
    }


def simulate_pennylane(
    qnode: Any,
    *,
    shots: int = 1024,
    noise_model: Optional[dict] = None,
) -> dict:
    """
    Execute a PennyLane QNode and return measurement results.

    Parameters
    ----------
    qnode : pennylane.QNode
        The QNode to execute.
    shots : int
        Number of shots (note: QNode should be created with shots parameter).
    noise_model : dict, optional
        Noise model specification. Currently only 'none' is supported;
        other types print a warning.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'samples': raw sample array
        - 'counts': measurement counts as {bitstring: count}
        - 'shots': number of shots
        - 'backend': 'pennylane'
    """
    if noise_model is not None and noise_model.get('type') != 'none':
        noise_type = noise_model.get('type', 'unknown')
        print(f"[sim_backends] Warning: Noise model '{noise_type}' not fully "
              f"implemented for PennyLane. Using ideal simulation.")

    # Execute the QNode
    samples = qnode()

    # Convert samples to counts
    if samples is not None and len(samples) > 0:
        # samples is a 2D array of shape (shots, n_qubits)
        counts = {}
        for sample in samples:
            bitstring = ''.join(str(int(b)) for b in sample)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        actual_shots = len(samples)
    else:
        counts = {}
        actual_shots = 0

    return {
        'samples': samples,
        'counts': counts,
        'shots': actual_shots,
        'backend': 'pennylane',
    }


def simulate_cirq(
    circuit: Any,
    *,
    shots: int = 1024,
    noise_model: Optional[dict] = None,
) -> dict:
    """
    Simulate a Cirq circuit and return measurement results.

    Parameters
    ----------
    circuit : cirq.Circuit
        The circuit to simulate.
    shots : int
        Number of measurement shots.
    noise_model : dict, optional
        Noise model specification. Currently only 'none' is supported;
        other types print a warning.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'counts': measurement counts as {bitstring: count}
        - 'shots': number of shots
        - 'backend': 'cirq'
    """
    try:
        import cirq
    except ImportError:
        raise ImportError(
            "Cirq is not installed. Install with: pip install cirq"
        )

    if noise_model is not None and noise_model.get('type') != 'none':
        noise_type = noise_model.get('type', 'unknown')
        print(f"[sim_backends] Warning: Noise model '{noise_type}' not fully "
              f"implemented for Cirq. Using ideal simulation.")

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)

    # Extract measurement results
    measurements = result.measurements.get('result', np.array([]))

    counts = {}
    for row in measurements:
        bitstring = ''.join(str(int(b)) for b in row)
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return {
        'counts': counts,
        'shots': shots,
        'backend': 'cirq',
    }


def simulate_circuit(
    circuit: Any,
    backend: str,
    *,
    shots: int = 1024,
    noise_model: Optional[dict] = None,
) -> dict:
    """
    Unified interface for simulating a circuit on any backend.

    Parameters
    ----------
    circuit : Any
        The circuit or QNode to simulate.
    backend : str
        Backend name: 'qiskit', 'pennylane', or 'cirq'.
    shots : int
        Number of measurement shots.
    noise_model : dict, optional
        Noise model specification.

    Returns
    -------
    results : dict
        Simulation results.

    Raises
    ------
    ValueError
        If backend is not recognized.
    """
    backend = backend.lower()

    if backend == 'qiskit':
        return simulate_qiskit(circuit, shots=shots, noise_model=noise_model)
    elif backend == 'pennylane':
        return simulate_pennylane(circuit, shots=shots, noise_model=noise_model)
    elif backend == 'cirq':
        return simulate_cirq(circuit, shots=shots, noise_model=noise_model)
    else:
        raise ValueError(f"Unknown backend: {backend}. Expected qiskit/pennylane/cirq.")
