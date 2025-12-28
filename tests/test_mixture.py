"""
Test suite for the TopologyMixture module.

Tests cover:
1. Softmax weight properties (sum to 1, non-negative)
2. Mixture mask shape and bounds
3. Gradient correctness via finite differences
4. Training convergence
"""

import unittest
import numpy as np
from sqnt_hardware_demo.graphs import make_graph_mask
from sqnt_hardware_demo.mixture import (
    TopologyMixture,
    softmax,
    compute_dL_dM,
    train_learned_mixture,
    train_fixed_topology,
    train_random_mixture,
)
from sqnt_hardware_demo.train_demo import make_synthetic


class TestSoftmax(unittest.TestCase):
    """Test suite for softmax function."""

    def test_softmax_sums_to_one(self):
        """Test that softmax outputs sum to 1."""
        z = np.array([1.0, 2.0, 3.0])
        w = softmax(z)
        self.assertAlmostEqual(np.sum(w), 1.0, places=10)

    def test_softmax_non_negative(self):
        """Test that softmax outputs are non-negative."""
        z = np.array([-10.0, 0.0, 10.0])
        w = softmax(z)
        self.assertTrue(np.all(w >= 0.0))

    def test_softmax_numerical_stability(self):
        """Test softmax with large values doesn't overflow."""
        z = np.array([1000.0, 1001.0, 1002.0])
        w = softmax(z)
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(np.sum(w), 1.0, places=10)

    def test_softmax_uniform_for_equal_inputs(self):
        """Test that equal inputs give uniform distribution."""
        z = np.array([5.0, 5.0, 5.0, 5.0])
        w = softmax(z)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(w, expected)


class TestTopologyMixture(unittest.TestCase):
    """Test suite for TopologyMixture class."""

    def setUp(self):
        """Set up test fixtures."""
        self.n = 4
        self.masks = [
            make_graph_mask("chain", self.n, include_self=True, normalize=True),
            make_graph_mask("ring", self.n, include_self=True, normalize=True),
        ]
        self.mixture = TopologyMixture(self.masks, seed=42)

    def test_initialization(self):
        """Test that mixture initializes correctly."""
        self.assertEqual(self.mixture.K, 2)
        self.assertEqual(self.mixture.n, 4)
        self.assertEqual(len(self.mixture.z), 2)

    def test_weights_sum_to_one(self):
        """Test that mixture weights sum to 1."""
        w = self.mixture.weights()
        self.assertAlmostEqual(np.sum(w), 1.0, places=10)

    def test_weights_non_negative(self):
        """Test that mixture weights are non-negative."""
        w = self.mixture.weights()
        self.assertTrue(np.all(w >= 0.0))

    def test_mixture_mask_shape(self):
        """Test that mixture mask has correct shape."""
        M = self.mixture.mixture_mask()
        self.assertEqual(M.shape, (self.n, self.n))

    def test_mixture_mask_bounded(self):
        """Test that mixture mask is bounded by component masks."""
        M = self.mixture.mixture_mask()
        M_min = np.minimum(self.masks[0], self.masks[1])
        M_max = np.maximum(self.masks[0], self.masks[1])

        # Mixture should be between min and max of components
        self.assertTrue(np.all(M >= M_min - 1e-10))
        self.assertTrue(np.all(M <= M_max + 1e-10))

    def test_step_updates_logits(self):
        """Test that step method updates logits."""
        z_before = self.mixture.z.copy()
        dL_dz = np.array([0.1, -0.1])
        lr = 0.5

        self.mixture.step(dL_dz, lr=lr)

        expected_z = z_before - lr * dL_dz
        np.testing.assert_array_almost_equal(self.mixture.z, expected_z)

    def test_deterministic_initialization(self):
        """Test that same seed gives same initialization."""
        mix1 = TopologyMixture(self.masks, seed=123)
        mix2 = TopologyMixture(self.masks, seed=123)
        np.testing.assert_array_equal(mix1.z, mix2.z)

    def test_different_seeds_differ(self):
        """Test that different seeds give different initializations."""
        mix1 = TopologyMixture(self.masks, seed=1)
        mix2 = TopologyMixture(self.masks, seed=2)
        self.assertFalse(np.allclose(mix1.z, mix2.z))


class TestGradientComputation(unittest.TestCase):
    """Test suite for gradient computation."""

    def setUp(self):
        """Set up test fixtures."""
        self.n = 4
        self.K = 2
        self.masks = [
            make_graph_mask("chain", self.n, include_self=True, normalize=True),
            make_graph_mask("complete", self.n, include_self=True, normalize=True),
        ]

    def test_compute_dL_dM_shape(self):
        """Test that dL/dM has correct shape."""
        W = np.random.randn(self.n, self.n)
        dL_dWeff = np.random.randn(self.n, self.n)
        dL_dM = compute_dL_dM(W, dL_dWeff)
        self.assertEqual(dL_dM.shape, (self.n, self.n))

    def test_grad_z_shape(self):
        """Test that gradient w.r.t. z has correct shape."""
        mixture = TopologyMixture(self.masks, seed=0)
        dL_dM = np.random.randn(self.n, self.n)
        dL_dz = mixture.grad_z(dL_dM)
        self.assertEqual(dL_dz.shape, (self.K,))

    def test_grad_z_finite(self):
        """Test that gradient is always finite."""
        mixture = TopologyMixture(self.masks, seed=0)
        dL_dM = np.random.randn(self.n, self.n) * 10
        dL_dz = mixture.grad_z(dL_dM)
        self.assertTrue(np.all(np.isfinite(dL_dz)))

    def test_gradient_finite_difference(self):
        """
        Numerical gradient check via finite differences.

        This is the key sanity check: analytic gradient should match
        the numerical gradient computed via central differences.
        """
        # Create a small test problem
        n = 4
        batch = 16
        rng = np.random.default_rng(42)

        masks = [
            make_graph_mask("chain", n, include_self=True, normalize=True),
            make_graph_mask("ring", n, include_self=True, normalize=True),
        ]
        mixture = TopologyMixture(masks, seed=0)

        # Random inputs
        X = rng.standard_normal((batch, n))
        y = rng.integers(0, 2, batch).astype(float)

        # Import layer
        from sqnt_hardware_demo.sqnt_layer import SQNTLayer

        model = SQNTLayer(n=n, seed=0)

        # Compute analytic gradient
        M = mixture.mixture_mask()
        W_eff = model.W * M
        p = model.forward(X, M)

        # Loss
        eps = 1e-9
        loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()

        # Gradients
        dlogits = (p - y) / batch
        h = X @ W_eff.T
        dH = np.outer(dlogits, model.v)
        dWeff = dH.T @ X
        dL_dM = compute_dL_dM(model.W, dWeff)
        analytic_grad = mixture.grad_z(dL_dM)

        # Compute numerical gradient via central differences
        delta = 1e-5
        numerical_grad = np.zeros(mixture.K)

        for k in range(mixture.K):
            # Perturb z[k] positively
            z_plus = mixture.z.copy()
            z_plus[k] += delta
            mixture.z = z_plus
            M_plus = mixture.mixture_mask()
            p_plus = model.forward(X, M_plus)
            loss_plus = -(y * np.log(p_plus + eps) + (1 - y) * np.log(1 - p_plus + eps)).mean()

            # Perturb z[k] negatively
            z_minus = mixture.z.copy()
            z_minus[k] -= 2 * delta  # go from z+delta to z-delta
            mixture.z = z_minus
            M_minus = mixture.mixture_mask()
            p_minus = model.forward(X, M_minus)
            loss_minus = -(y * np.log(p_minus + eps) + (1 - y) * np.log(1 - p_minus + eps)).mean()

            # Central difference
            numerical_grad[k] = (loss_plus - loss_minus) / (2 * delta)

            # Reset z
            mixture.z = z_plus - delta  # back to original

        # Check agreement
        np.testing.assert_array_almost_equal(
            analytic_grad, numerical_grad, decimal=4,
            err_msg="Analytic gradient does not match numerical gradient"
        )


class TestTrainLearnedMixture(unittest.TestCase):
    """Test suite for train_learned_mixture function."""

    def test_returns_valid_accuracy(self):
        """Test that training returns valid accuracy."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        acc, hist = train_learned_mixture(
            X, y,
            topology_names=["chain", "ring"],
            n=n,
            epochs=10,
            seed=0,
        )
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_returns_history(self):
        """Test that training returns proper history dict."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        acc, hist = train_learned_mixture(
            X, y,
            topology_names=["chain", "ring"],
            n=n,
            epochs=10,
            seed=0,
        )
        self.assertIn('loss', hist)
        self.assertIn('acc', hist)
        self.assertIn('weights', hist)
        self.assertEqual(len(hist['loss']), 10)
        self.assertEqual(len(hist['acc']), 10)
        self.assertEqual(len(hist['weights']), 10)

    def test_weights_history_valid(self):
        """Test that weight history contains valid weights."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        acc, hist = train_learned_mixture(
            X, y,
            topology_names=["chain", "ring", "star"],
            n=n,
            epochs=10,
            seed=0,
        )
        for w in hist['weights']:
            self.assertAlmostEqual(np.sum(w), 1.0, places=8)
            self.assertTrue(np.all(w >= 0.0))
            self.assertEqual(len(w), 3)

    def test_deterministic(self):
        """Test that same seed gives same results."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        acc1, _ = train_learned_mixture(
            X, y, topology_names=["chain", "ring"], n=n, epochs=20, seed=42
        )
        acc2, _ = train_learned_mixture(
            X, y, topology_names=["chain", "ring"], n=n, epochs=20, seed=42
        )
        self.assertAlmostEqual(acc1, acc2, places=10)


class TestTrainFixedTopology(unittest.TestCase):
    """Test suite for train_fixed_topology function."""

    def test_returns_valid_accuracy(self):
        """Test that training returns valid accuracy."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        acc, hist = train_fixed_topology(
            X, y,
            topology_name="ring",
            n=n,
            epochs=10,
            seed=0,
        )
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_returns_history(self):
        """Test that training returns proper history dict."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        acc, hist = train_fixed_topology(
            X, y,
            topology_name="chain",
            n=n,
            epochs=10,
            seed=0,
        )
        self.assertIn('loss', hist)
        self.assertIn('acc', hist)
        self.assertEqual(len(hist['loss']), 10)

    def test_different_topologies(self):
        """Test training works for all topology types."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        for topo in ["chain", "ring", "star", "complete"]:
            acc, _ = train_fixed_topology(
                X, y, topology_name=topo, n=n, epochs=5, seed=0
            )
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)


class TestTrainRandomMixture(unittest.TestCase):
    """Test suite for train_random_mixture function."""

    def test_returns_valid_accuracy(self):
        """Test that training returns valid accuracy."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        acc, hist, weights = train_random_mixture(
            X, y,
            topology_names=["chain", "ring"],
            n=n,
            epochs=10,
            seed=0,
        )
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_returns_frozen_weights(self):
        """Test that frozen weights are valid."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        acc, hist, weights = train_random_mixture(
            X, y,
            topology_names=["chain", "ring", "star"],
            n=n,
            epochs=10,
            seed=0,
        )
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=8)
        self.assertTrue(np.all(weights >= 0.0))

    def test_frozen_weights_deterministic(self):
        """Test that same seed gives same frozen weights."""
        n = 6
        X, y = make_synthetic(n=n, batch=64, seed=0)
        _, _, w1 = train_random_mixture(
            X, y, topology_names=["chain", "ring"], n=n, epochs=5, seed=42
        )
        _, _, w2 = train_random_mixture(
            X, y, topology_names=["chain", "ring"], n=n, epochs=5, seed=42
        )
        np.testing.assert_array_almost_equal(w1, w2)


if __name__ == '__main__':
    unittest.main()
