"""
Quantum circuit compilation targets for SQNT topologies.

Provides minimal circuit builders for Qiskit, PennyLane, and Cirq backends.
These functions convert topology masks into quantum circuits for
compilation and simulation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def build_observable_set(n: int) -> dict:
    """
    Define a minimal set of observables for testing.

    Returns a dictionary specifying measurement bases and observable
    operators for use across different backends.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    observables : dict
        Observable specification with keys:
        - 'paulis': list of Pauli strings to measure
        - 'qubits': qubit indices for each observable
    """
    observables = {
        'paulis': [],
        'qubits': [],
    }

    # Single-qubit Z measurements
    for i in range(n):
        observables['paulis'].append(f'Z{i}')
        observables['qubits'].append([i])

    # Two-qubit ZZ on adjacent pairs
    for i in range(n - 1):
        observables['paulis'].append(f'Z{i}Z{i+1}')
        observables['qubits'].append([i, i + 1])

    return observables


def _mask_to_edges(mask: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, int]]:
    """
    Extract edges from a topology mask.

    Parameters
    ----------
    mask : np.ndarray
        Topology mask of shape (n, n).
    threshold : float
        Minimum mask value to consider as an edge.

    Returns
    -------
    edges : List[Tuple[int, int]]
        List of (i, j) pairs with i < j.
    """
    n = mask.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (mask[i, j] + mask[j, i]) / 2.0 > threshold:
                edges.append((i, j))
    return edges


def to_qiskit_circuit(
    mask: np.ndarray,
    *,
    gate_set: str = "cx,rz",
    seed: int = 0,
) -> Any:
    """
    Convert a topology mask to a Qiskit QuantumCircuit.

    Creates a simple parameterized circuit where edges in the mask
    become two-qubit gates (CNOT + RZ).

    Parameters
    ----------
    mask : np.ndarray
        Topology mask of shape (n, n).
    gate_set : str
        Gate set to use. Currently supports "cx,rz".
    seed : int
        Random seed for deterministic angle generation.

    Returns
    -------
    circuit : qiskit.QuantumCircuit
        The constructed quantum circuit.

    Raises
    ------
    ImportError
        If qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError(
            "Qiskit is not installed. Install with: pip install qiskit"
        )

    n = mask.shape[0]
    edges = _mask_to_edges(mask)

    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=len(edges) * 2)

    qc = QuantumCircuit(n, name="sqnt_topology")

    # Initial Hadamard layer
    for i in range(n):
        qc.h(i)

    # Apply two-qubit interactions for each edge
    for idx, (i, j) in enumerate(edges):
        qc.cx(i, j)
        qc.rz(angles[2 * idx], j)
        qc.cx(i, j)
        qc.rz(angles[2 * idx + 1], i)

    # Measurement layer
    qc.measure_all()

    return qc


def to_pennylane_qnode(
    mask: np.ndarray,
    *,
    shots: int = 1024,
    seed: int = 0,
) -> Any:
    """
    Create a PennyLane QNode from a topology mask.

    The QNode implements a circuit with two-qubit gates corresponding
    to edges in the mask.

    Parameters
    ----------
    mask : np.ndarray
        Topology mask of shape (n, n).
    shots : int
        Number of shots for sampling.
    seed : int
        Random seed for deterministic angle generation.

    Returns
    -------
    qnode : pennylane.QNode
        The constructed QNode.

    Raises
    ------
    ImportError
        If pennylane is not installed.
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError(
            "PennyLane is not installed. Install with: pip install pennylane"
        )

    n = mask.shape[0]
    edges = _mask_to_edges(mask)

    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=len(edges) * 2)

    dev = qml.device("default.qubit", wires=n, shots=shots)

    @qml.qnode(dev)
    def circuit():
        # Initial Hadamard layer
        for i in range(n):
            qml.Hadamard(wires=i)

        # Two-qubit interactions
        for idx, (i, j) in enumerate(edges):
            qml.CNOT(wires=[i, j])
            qml.RZ(angles[2 * idx], wires=j)
            qml.CNOT(wires=[i, j])
            qml.RZ(angles[2 * idx + 1], wires=i)

        # Return computational basis samples
        return qml.sample()

    return circuit


def to_cirq_circuit(
    mask: np.ndarray,
    *,
    seed: int = 0,
) -> Any:
    """
    Convert a topology mask to a Cirq Circuit.

    Creates a circuit with two-qubit gates corresponding to edges
    in the mask.

    Parameters
    ----------
    mask : np.ndarray
        Topology mask of shape (n, n).
    seed : int
        Random seed for deterministic angle generation.

    Returns
    -------
    circuit : cirq.Circuit
        The constructed circuit.

    Raises
    ------
    ImportError
        If cirq is not installed.
    """
    try:
        import cirq
    except ImportError:
        raise ImportError(
            "Cirq is not installed. Install with: pip install cirq"
        )

    n = mask.shape[0]
    edges = _mask_to_edges(mask)

    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=len(edges) * 2)

    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()

    # Initial Hadamard layer
    circuit.append([cirq.H(q) for q in qubits])

    # Two-qubit interactions
    for idx, (i, j) in enumerate(edges):
        circuit.append(cirq.CNOT(qubits[i], qubits[j]))
        circuit.append(cirq.rz(angles[2 * idx])(qubits[j]))
        circuit.append(cirq.CNOT(qubits[i], qubits[j]))
        circuit.append(cirq.rz(angles[2 * idx + 1])(qubits[i]))

    # Measurement
    circuit.append(cirq.measure(*qubits, key='result'))

    return circuit


def check_backend_available(backend: str) -> bool:
    """
    Check if a quantum backend is available.

    Parameters
    ----------
    backend : str
        Backend name: 'qiskit', 'pennylane', or 'cirq'.

    Returns
    -------
    available : bool
        True if the backend can be imported.
    """
    backend = backend.lower()
    if backend == 'qiskit':
        try:
            import qiskit
            return True
        except ImportError:
            return False
    elif backend == 'pennylane':
        try:
            import pennylane
            return True
        except ImportError:
            return False
    elif backend == 'cirq':
        try:
            import cirq
            return True
        except ImportError:
            return False
    else:
        return False
