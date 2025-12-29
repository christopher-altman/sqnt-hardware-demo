"""
Test suite for Phase III: Compilation & Simulation Integration.

Covers:
1. Device graph loading + adjacency shape
2. Routing/SWAP cost proxy
3. Compilation penalty & gradient
4. Mixture recovery integration
5. Backend smoke tests (with skip markers for optional deps)
"""

import unittest
import tempfile
import json
import os
import numpy as np

from sqnt_hardware_demo.graphs import make_graph_mask


# =============================================================================
# Test Device Graph Loading
# =============================================================================

class TestDeviceGraphLoading(unittest.TestCase):
    """Test device graph loading and adjacency conversion."""

    def setUp(self):
        """Create temporary device graph files."""
        self.temp_dir = tempfile.mkdtemp()

        # Valid device graph
        self.valid_graph = {
            "name": "test_graph",
            "nodes": [0, 1, 2, 3],
            "edges": [[0, 1], [1, 2], [2, 3]],
            "metadata": {"connectivity": "linear"}
        }
        self.valid_path = os.path.join(self.temp_dir, "valid.json")
        with open(self.valid_path, 'w') as f:
            json.dump(self.valid_graph, f)

        # Missing nodes key
        self.missing_nodes = {"edges": [[0, 1]]}
        self.missing_nodes_path = os.path.join(self.temp_dir, "missing_nodes.json")
        with open(self.missing_nodes_path, 'w') as f:
            json.dump(self.missing_nodes, f)

        # Missing edges key
        self.missing_edges = {"nodes": [0, 1]}
        self.missing_edges_path = os.path.join(self.temp_dir, "missing_edges.json")
        with open(self.missing_edges_path, 'w') as f:
            json.dump(self.missing_edges, f)

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_valid_device_graph(self):
        """Test loading a valid device graph."""
        from sqnt_hardware_demo.compilation import load_device_graph

        graph = load_device_graph(self.valid_path)
        self.assertEqual(graph['name'], 'test_graph')
        self.assertEqual(len(graph['nodes']), 4)
        self.assertEqual(len(graph['edges']), 3)

    def test_load_missing_file_raises(self):
        """Test that missing file raises FileNotFoundError."""
        from sqnt_hardware_demo.compilation import load_device_graph

        with self.assertRaises(FileNotFoundError):
            load_device_graph("/nonexistent/path.json")

    def test_load_missing_nodes_infers_from_edges(self):
        """Test that missing 'nodes' key is auto-inferred from edges."""
        from sqnt_hardware_demo.compilation import load_device_graph

        # Should now auto-populate nodes from edges
        graph = load_device_graph(self.missing_nodes_path)
        self.assertIn("nodes", graph)
        # Should have inferred nodes [0, 1] from edge [[0, 1]]
        self.assertEqual(sorted(graph["nodes"]), [0, 1])

    def test_load_missing_edges_raises(self):
        """Test that missing 'edges' key raises KeyError."""
        from sqnt_hardware_demo.compilation import load_device_graph

        with self.assertRaises(KeyError) as ctx:
            load_device_graph(self.missing_edges_path)
        self.assertIn("edges", str(ctx.exception))

    def test_device_graph_to_adjacency_shape(self):
        """Test adjacency matrix has correct shape."""
        from sqnt_hardware_demo.compilation import (
            load_device_graph,
            device_graph_to_adjacency
        )

        graph = load_device_graph(self.valid_path)
        n = 6  # Larger than device graph
        adj = device_graph_to_adjacency(graph, n)

        self.assertEqual(adj.shape, (n, n))

    def test_device_graph_to_adjacency_symmetric(self):
        """Test adjacency matrix is symmetric."""
        from sqnt_hardware_demo.compilation import (
            load_device_graph,
            device_graph_to_adjacency
        )

        graph = load_device_graph(self.valid_path)
        adj = device_graph_to_adjacency(graph, 4)

        np.testing.assert_array_equal(adj, adj.T)

    def test_device_graph_to_adjacency_values(self):
        """Test adjacency matrix has correct edge values."""
        from sqnt_hardware_demo.compilation import (
            load_device_graph,
            device_graph_to_adjacency
        )

        graph = load_device_graph(self.valid_path)
        adj = device_graph_to_adjacency(graph, 4)

        # Check edges exist
        self.assertEqual(adj[0, 1], 1.0)
        self.assertEqual(adj[1, 0], 1.0)
        self.assertEqual(adj[1, 2], 1.0)
        self.assertEqual(adj[2, 3], 1.0)

        # Check non-edges
        self.assertEqual(adj[0, 2], 0.0)
        self.assertEqual(adj[0, 3], 0.0)


# =============================================================================
# Test Routing/SWAP Cost
# =============================================================================

class TestRoutingSwapCost(unittest.TestCase):
    """Test routing/SWAP cost proxy."""

    def setUp(self):
        """Create test masks and device adjacency."""
        self.n = 4
        # Linear device: 0-1-2-3
        self.device_adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)

    def test_cost_zero_for_subset(self):
        """Test cost is 0 when mask edges are subset of device."""
        from sqnt_hardware_demo.compilation import routing_swap_cost

        # Chain mask matches linear device
        mask = make_graph_mask("chain", self.n, include_self=False, normalize=False)
        cost = routing_swap_cost(mask, self.device_adj)

        self.assertEqual(cost, 0.0)

    def test_cost_increases_for_non_native(self):
        """Test cost increases for non-native edges."""
        from sqnt_hardware_demo.compilation import routing_swap_cost

        # Complete graph has edges not in linear device
        mask = make_graph_mask("complete", self.n, include_self=False, normalize=False)
        cost = routing_swap_cost(mask, self.device_adj)

        self.assertGreater(cost, 0.0)

    def test_cost_deterministic(self):
        """Test cost is deterministic across runs."""
        from sqnt_hardware_demo.compilation import routing_swap_cost

        mask = make_graph_mask("star", self.n, include_self=False, normalize=False)

        cost1 = routing_swap_cost(mask, self.device_adj)
        cost2 = routing_swap_cost(mask, self.device_adj)

        self.assertEqual(cost1, cost2)

    def test_cost_finite(self):
        """Test cost is finite."""
        from sqnt_hardware_demo.compilation import routing_swap_cost

        mask = make_graph_mask("complete", self.n, include_self=True, normalize=True)
        cost = routing_swap_cost(mask, self.device_adj)

        self.assertTrue(np.isfinite(cost))


# =============================================================================
# Test Compilation Penalty & Gradient
# =============================================================================

class TestCompilationPenalty(unittest.TestCase):
    """Test compilation penalty and gradient."""

    def setUp(self):
        """Create test fixtures."""
        self.n = 4
        self.masks = [
            make_graph_mask("chain", self.n, include_self=True, normalize=True),
            make_graph_mask("star", self.n, include_self=True, normalize=True),
            make_graph_mask("complete", self.n, include_self=True, normalize=True),
        ]
        self.K = len(self.masks)
        self.weights = np.array([0.5, 0.3, 0.2])

        # Linear device graph
        self.device_graph = {
            "name": "linear",
            "nodes": [0, 1, 2, 3],
            "edges": [[0, 1], [1, 2], [2, 3]],
        }

    def test_penalty_zero_when_lambda_zero(self):
        """Test penalty is 0 when lambda_compile = 0."""
        from sqnt_hardware_demo.compilation import compilation_penalty

        penalty = compilation_penalty(
            self.weights, self.masks, self.device_graph, lambda_compile=0.0
        )

        self.assertEqual(penalty, 0.0)

    def test_penalty_increases_with_lambda(self):
        """Test penalty increases with lambda_compile."""
        from sqnt_hardware_demo.compilation import compilation_penalty

        penalty_low = compilation_penalty(
            self.weights, self.masks, self.device_graph, lambda_compile=0.1
        )
        penalty_high = compilation_penalty(
            self.weights, self.masks, self.device_graph, lambda_compile=1.0
        )

        self.assertLess(penalty_low, penalty_high)

    def test_grad_zero_when_lambda_zero(self):
        """Test gradient is 0 when lambda_compile = 0."""
        from sqnt_hardware_demo.compilation import compilation_penalty_grad

        grad = compilation_penalty_grad(
            self.weights, self.masks, self.device_graph, lambda_compile=0.0
        )

        np.testing.assert_array_equal(grad, np.zeros(self.K))

    def test_grad_correct_shape(self):
        """Test gradient has correct shape."""
        from sqnt_hardware_demo.compilation import compilation_penalty_grad

        grad = compilation_penalty_grad(
            self.weights, self.masks, self.device_graph, lambda_compile=0.1
        )

        self.assertEqual(grad.shape, (self.K,))

    def test_grad_logits_correct_shape(self):
        """Test logit gradient has correct shape."""
        from sqnt_hardware_demo.compilation import compilation_penalty_grad_logits

        grad_z = compilation_penalty_grad_logits(
            self.weights, self.masks, self.device_graph, lambda_compile=0.1
        )

        self.assertEqual(grad_z.shape, (self.K,))

    def test_grad_pushes_toward_low_cost(self):
        """Test gradient encourages low-cost topologies."""
        from sqnt_hardware_demo.compilation import (
            compilation_penalty_grad,
            routing_swap_cost,
            device_graph_to_adjacency,
        )

        device_adj = device_graph_to_adjacency(self.device_graph, self.n)

        # Compute costs per topology
        costs = np.array([
            routing_swap_cost(mask, device_adj) for mask in self.masks
        ])

        grad = compilation_penalty_grad(
            self.weights, self.masks, self.device_graph, lambda_compile=0.1
        )

        # Gradient should be higher for higher-cost topologies
        # (chain has lowest cost, complete has highest)
        self.assertLess(grad[0], grad[2])  # chain < complete


# =============================================================================
# Test Mixture Recovery Integration
# =============================================================================

class TestMixtureRecoveryIntegration(unittest.TestCase):
    """Test mixture recovery with compilation constraints."""

    def setUp(self):
        """Create test data."""
        from sqnt_hardware_demo.experiments import (
            sample_ground_truth_mixture,
            generate_planted_mixture_data,
        )

        self.n = 8
        self.topologies = ["chain", "ring", "star", "complete"]
        self.K = len(self.topologies)
        self.w_true = sample_ground_truth_mixture(self.K, seed=0, concentration=0.3)

        self.X, self.y, _, _ = generate_planted_mixture_data(
            n=self.n,
            batch=128,
            w_true=self.w_true,
            topology_names=self.topologies,
            seed=0,
            noise_level=0.05,
        )

        # Linear device graph
        self.device_graph = {
            "name": "linear",
            "nodes": list(range(self.n)),
            "edges": [[i, i + 1] for i in range(self.n - 1)],
        }

    def test_baseline_unchanged_without_constraints(self):
        """Test behavior is unchanged when compile constraints are off."""
        from sqnt_hardware_demo.experiments import train_mixture_recovery

        history = train_mixture_recovery(
            self.X, self.y, self.w_true,
            topology_names=self.topologies,
            n=self.n,
            epochs=10,
            seed=0,
            enable_compile_constraints=False,
        )

        self.assertNotIn('loss_compile', history)
        self.assertEqual(len(history['loss']), 10)

    def test_compile_penalty_tracked_when_enabled(self):
        """Test compilation penalty is tracked when enabled."""
        from sqnt_hardware_demo.experiments import train_mixture_recovery

        history = train_mixture_recovery(
            self.X, self.y, self.w_true,
            topology_names=self.topologies,
            n=self.n,
            epochs=10,
            seed=0,
            enable_compile_constraints=True,
            device_graph=self.device_graph,
            lambda_compile=0.1,
        )

        self.assertIn('loss_compile', history)
        self.assertEqual(len(history['loss_compile']), 10)

    def test_compile_penalty_zero_when_lambda_zero(self):
        """Test no penalty when lambda is 0 even if enabled."""
        from sqnt_hardware_demo.experiments import train_mixture_recovery

        history = train_mixture_recovery(
            self.X, self.y, self.w_true,
            topology_names=self.topologies,
            n=self.n,
            epochs=10,
            seed=0,
            enable_compile_constraints=True,
            device_graph=self.device_graph,
            lambda_compile=0.0,  # Lambda is 0
        )

        # Should not track compile loss when lambda is 0
        self.assertNotIn('loss_compile', history)

    def test_compile_penalty_requires_device_graph(self):
        """Test no penalty when device graph is None."""
        from sqnt_hardware_demo.experiments import train_mixture_recovery

        history = train_mixture_recovery(
            self.X, self.y, self.w_true,
            topology_names=self.topologies,
            n=self.n,
            epochs=10,
            seed=0,
            enable_compile_constraints=True,
            device_graph=None,  # No device graph
            lambda_compile=0.1,
        )

        self.assertNotIn('loss_compile', history)


# =============================================================================
# Test Backend Smoke Tests
# =============================================================================

def _check_qiskit_available():
    """Check if Qiskit is available."""
    try:
        import qiskit
        return True
    except ImportError:
        return False


def _check_pennylane_available():
    """Check if PennyLane is available."""
    try:
        import pennylane
        return True
    except ImportError:
        return False


def _check_cirq_available():
    """Check if Cirq is available."""
    try:
        import cirq
        return True
    except ImportError:
        return False


class TestBackendAvailability(unittest.TestCase):
    """Test backend availability checking."""

    def test_check_backend_available(self):
        """Test check_backend_available function."""
        from sqnt_hardware_demo.circuit_targets import check_backend_available

        # Check returns bool for valid backends
        for backend in ['qiskit', 'pennylane', 'cirq']:
            result = check_backend_available(backend)
            self.assertIsInstance(result, bool)

        # Unknown backend returns False
        self.assertFalse(check_backend_available('unknown_backend'))


class TestObservableSet(unittest.TestCase):
    """Test observable set builder."""

    def test_build_observable_set(self):
        """Test building observable set."""
        from sqnt_hardware_demo.circuit_targets import build_observable_set

        n = 4
        obs = build_observable_set(n)

        self.assertIn('paulis', obs)
        self.assertIn('qubits', obs)
        self.assertEqual(len(obs['paulis']), n + (n - 1))  # Z's + ZZ's


@unittest.skipUnless(_check_qiskit_available(), "Qiskit not installed")
class TestQiskitBackend(unittest.TestCase):
    """Test Qiskit circuit compilation and simulation."""

    def test_to_qiskit_circuit(self):
        """Test Qiskit circuit creation."""
        from sqnt_hardware_demo.circuit_targets import to_qiskit_circuit

        mask = make_graph_mask("ring", 4, include_self=False, normalize=False)
        circuit = to_qiskit_circuit(mask, seed=0)

        self.assertEqual(circuit.num_qubits, 4)

    def test_qiskit_simulation(self):
        """Test Qiskit simulation."""
        from sqnt_hardware_demo.circuit_targets import to_qiskit_circuit
        from sqnt_hardware_demo.sim_backends import simulate_qiskit

        mask = make_graph_mask("chain", 3, include_self=False, normalize=False)
        circuit = to_qiskit_circuit(mask, seed=0)
        results = simulate_qiskit(circuit, shots=100)

        self.assertIn('counts', results)
        self.assertEqual(results['backend'], 'qiskit')


@unittest.skipUnless(_check_pennylane_available(), "PennyLane not installed")
class TestPennyLaneBackend(unittest.TestCase):
    """Test PennyLane QNode creation and simulation."""

    def test_to_pennylane_qnode(self):
        """Test PennyLane QNode creation."""
        from sqnt_hardware_demo.circuit_targets import to_pennylane_qnode

        mask = make_graph_mask("star", 4, include_self=False, normalize=False)
        qnode = to_pennylane_qnode(mask, shots=100, seed=0)

        self.assertIsNotNone(qnode)

    def test_pennylane_simulation(self):
        """Test PennyLane simulation."""
        from sqnt_hardware_demo.circuit_targets import to_pennylane_qnode
        from sqnt_hardware_demo.sim_backends import simulate_pennylane

        mask = make_graph_mask("chain", 3, include_self=False, normalize=False)
        qnode = to_pennylane_qnode(mask, shots=100, seed=0)
        results = simulate_pennylane(qnode, shots=100)

        self.assertIn('counts', results)
        self.assertEqual(results['backend'], 'pennylane')


@unittest.skipUnless(_check_cirq_available(), "Cirq not installed")
class TestCirqBackend(unittest.TestCase):
    """Test Cirq circuit creation and simulation."""

    def test_to_cirq_circuit(self):
        """Test Cirq circuit creation."""
        from sqnt_hardware_demo.circuit_targets import to_cirq_circuit

        mask = make_graph_mask("complete", 3, include_self=False, normalize=False)
        circuit = to_cirq_circuit(mask, seed=0)

        self.assertIsNotNone(circuit)

    def test_cirq_simulation(self):
        """Test Cirq simulation."""
        from sqnt_hardware_demo.circuit_targets import to_cirq_circuit
        from sqnt_hardware_demo.sim_backends import simulate_cirq

        mask = make_graph_mask("chain", 3, include_self=False, normalize=False)
        circuit = to_cirq_circuit(mask, seed=0)
        results = simulate_cirq(circuit, shots=100)

        self.assertIn('counts', results)
        self.assertEqual(results['backend'], 'cirq')


class TestBackendGracefulSkip(unittest.TestCase):
    """Test graceful handling when backends are missing."""

    def test_qiskit_raises_import_error(self):
        """Test Qiskit raises ImportError when not installed."""
        if _check_qiskit_available():
            self.skipTest("Qiskit is installed; cannot test missing case")

        from sqnt_hardware_demo.circuit_targets import to_qiskit_circuit

        mask = make_graph_mask("chain", 3, include_self=False, normalize=False)
        with self.assertRaises(ImportError):
            to_qiskit_circuit(mask)

    def test_pennylane_raises_import_error(self):
        """Test PennyLane raises ImportError when not installed."""
        if _check_pennylane_available():
            self.skipTest("PennyLane is installed; cannot test missing case")

        from sqnt_hardware_demo.circuit_targets import to_pennylane_qnode

        mask = make_graph_mask("chain", 3, include_self=False, normalize=False)
        with self.assertRaises(ImportError):
            to_pennylane_qnode(mask)

    def test_cirq_raises_import_error(self):
        """Test Cirq raises ImportError when not installed."""
        if _check_cirq_available():
            self.skipTest("Cirq is installed; cannot test missing case")

        from sqnt_hardware_demo.circuit_targets import to_cirq_circuit

        mask = make_graph_mask("chain", 3, include_self=False, normalize=False)
        with self.assertRaises(ImportError):
            to_cirq_circuit(mask)


# =============================================================================
# Test Noise Model Loading
# =============================================================================

class TestNoiseModelLoading(unittest.TestCase):
    """Test noise model loading."""

    def setUp(self):
        """Create temp noise model files."""
        self.temp_dir = tempfile.mkdtemp()

        self.noise_none = {"type": "none", "parameters": {}}
        self.noise_path = os.path.join(self.temp_dir, "noise.json")
        with open(self.noise_path, 'w') as f:
            json.dump(self.noise_none, f)

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_noise_model(self):
        """Test loading noise model JSON."""
        from sqnt_hardware_demo.sim_backends import load_noise_model

        model = load_noise_model(self.noise_path)
        self.assertEqual(model['type'], 'none')


if __name__ == '__main__':
    unittest.main()
