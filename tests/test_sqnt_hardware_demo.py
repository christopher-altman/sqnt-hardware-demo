import unittest
import numpy as np
from sqnt_hardware_demo.graphs import make_graph_mask
from sqnt_hardware_demo.sqnt_layer import SQNTLayer, sigmoid
from sqnt_hardware_demo.train_demo import train_for_alpha, sweep_alphas, make_synthetic


class TestMakeGraphMask(unittest.TestCase):
    """Test suite for make_graph_mask function."""

    def test_chain_no_self_no_norm(self):
        """Test chain graph without self-loops and normalization."""
        M = make_graph_mask("chain", n=4, include_self=False, normalize=False)
        expected = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=float)
        np.testing.assert_array_equal(M, expected)

    def test_chain_with_self_no_norm(self):
        """Test chain graph with self-loops but no normalization."""
        M = make_graph_mask("chain", n=3, include_self=True, normalize=False)
        expected = np.array([
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1]
        ], dtype=float)
        np.testing.assert_array_equal(M, expected)

    def test_chain_with_self_and_norm(self):
        """Test chain graph with self-loops and normalization."""
        M = make_graph_mask("chain", n=3, include_self=True, normalize=True)
        expected = np.array([
            [0.5, 0.5, 0.0],
            [1/3, 1/3, 1/3],
            [0.0, 0.5, 0.5]
        ], dtype=float)
        np.testing.assert_array_almost_equal(M, expected)

    def test_ring_no_self_no_norm(self):
        """Test ring graph without self-loops and normalization."""
        M = make_graph_mask("ring", n=4, include_self=False, normalize=False)
        expected = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=float)
        np.testing.assert_array_equal(M, expected)

    def test_ring_with_norm(self):
        """Test ring graph with normalization."""
        M = make_graph_mask("ring", n=4, include_self=False, normalize=True)
        expected = np.array([
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0.0]
        ], dtype=float)
        np.testing.assert_array_almost_equal(M, expected)

    def test_star_no_self_no_norm(self):
        """Test star graph without self-loops and normalization."""
        M = make_graph_mask("star", n=5, include_self=False, normalize=False)
        expected = np.array([
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ], dtype=float)
        np.testing.assert_array_equal(M, expected)

    def test_star_with_self_and_norm(self):
        """Test star graph with self-loops and normalization."""
        M = make_graph_mask("star", n=3, include_self=True, normalize=True)
        # Hub (node 0) has connections to nodes 1 and 2, plus itself = 3
        # Nodes 1 and 2 each connect to hub and themselves = 2 each
        expected = np.array([
            [1/3, 1/3, 1/3],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5]
        ], dtype=float)
        np.testing.assert_array_almost_equal(M, expected)

    def test_complete_no_self_no_norm(self):
        """Test complete graph without self-loops and normalization."""
        M = make_graph_mask("complete", n=3, include_self=False, normalize=False)
        expected = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=float)
        np.testing.assert_array_equal(M, expected)

    def test_complete_with_self_and_norm(self):
        """Test complete graph with self-loops and normalization."""
        M = make_graph_mask("complete", n=3, include_self=True, normalize=True)
        expected = np.array([
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3]
        ], dtype=float)
        np.testing.assert_array_almost_equal(M, expected)

    def test_invalid_kind_raises_error(self):
        """Test that invalid graph kind raises ValueError."""
        with self.assertRaises(ValueError) as context:
            make_graph_mask("invalid", n=3)
        self.assertIn("Unknown kind", str(context.exception))

    def test_case_insensitive_kind(self):
        """Test that graph kind is case-insensitive."""
        M1 = make_graph_mask("CHAIN", n=3, include_self=False, normalize=False)
        M2 = make_graph_mask("chain", n=3, include_self=False, normalize=False)
        np.testing.assert_array_equal(M1, M2)


class TestSQNTLayer(unittest.TestCase):
    """Test suite for SQNTLayer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.n = 4
        self.layer = SQNTLayer(n=self.n, seed=42)
        self.X = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.M = np.ones((self.n, self.n)) / self.n

    def test_layer_initialization(self):
        """Test that layer initializes with correct shape and seed."""
        layer1 = SQNTLayer(n=5, seed=100)
        layer2 = SQNTLayer(n=5, seed=100)
        
        self.assertEqual(layer1.W.shape, (5, 5))
        self.assertEqual(layer1.v.shape, (5,))
        np.testing.assert_array_equal(layer1.W, layer2.W)
        np.testing.assert_array_equal(layer1.v, layer2.v)

    def test_forward_output_shape(self):
        """Test that forward produces correct output shape."""
        X = np.random.randn(10, self.n)
        M = np.eye(self.n)
        output = self.layer.forward(X, M)
        self.assertEqual(output.shape, (10,))

    def test_forward_output_range(self):
        """Test that forward output is in valid probability range [0, 1]."""
        X = np.random.randn(20, self.n)
        M = np.eye(self.n)
        output = self.layer.forward(X, M)
        self.assertTrue(np.all(output >= 0.0))
        self.assertTrue(np.all(output <= 1.0))

    def test_forward_with_identity_mask(self):
        """Test forward pass with identity mask."""
        M = np.eye(self.n)
        output = self.layer.forward(self.X, M)
        
        # With identity mask, W_eff = W * I = diag(W)
        W_eff = self.layer.W * M
        h = self.X @ W_eff.T
        expected_logits = h @ self.layer.v
        expected = sigmoid(expected_logits)
        
        np.testing.assert_array_almost_equal(output, expected)

    def test_forward_with_zero_mask(self):
        """Test forward pass with zero mask."""
        M = np.zeros((self.n, self.n))
        output = self.layer.forward(self.X, M)
        
        # With zero mask, W_eff = 0, so h = 0, logits = 0
        # sigmoid(0) = 0.5
        expected = np.full(self.X.shape[0], 0.5)
        np.testing.assert_array_almost_equal(output, expected)

    def test_forward_deterministic(self):
        """Test that forward is deterministic for same inputs."""
        M = np.random.randn(self.n, self.n)
        out1 = self.layer.forward(self.X, M)
        out2 = self.layer.forward(self.X, M)
        np.testing.assert_array_equal(out1, out2)

    def test_loss_and_grads_shapes(self):
        """Test that loss_and_grads returns correct shapes."""
        X = np.random.randn(50, self.n)
        y = np.random.randint(0, 2, 50).astype(float)
        M = np.eye(self.n)
        
        loss, dW, dv = self.layer.loss_and_grads(X, y, M)
        
        self.assertIsInstance(loss, (float, np.floating))
        self.assertEqual(dW.shape, (self.n, self.n))
        self.assertEqual(dv.shape, (self.n,))

    def test_loss_and_grads_positive_loss(self):
        """Test that loss is non-negative."""
        X = np.random.randn(30, self.n)
        y = np.random.randint(0, 2, 30).astype(float)
        M = np.eye(self.n)
        
        loss, _, _ = self.layer.loss_and_grads(X, y, M)
        self.assertGreaterEqual(loss, 0.0)

    def test_loss_and_grads_perfect_prediction(self):
        """Test loss is reasonable for well-separated predictions."""
        # Create a simple scenario where predictions should be good
        layer = SQNTLayer(n=2, seed=0)
        layer.W = np.array([[10.0, 0.0], [0.0, -10.0]])
        layer.v = np.array([1.0, 0.0])
        
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([1.0, 0.0])
        M = np.eye(2)
        
        loss, _, _ = layer.loss_and_grads(X, y, M)
        # Loss should be finite and non-negative
        self.assertGreaterEqual(loss, 0.0)
        self.assertTrue(np.isfinite(loss))

    def test_loss_and_grads_gradient_finite(self):
        """Test that gradients are finite (no NaN or Inf)."""
        X = np.random.randn(40, self.n)
        y = np.random.randint(0, 2, 40).astype(float)
        M = np.random.randn(self.n, self.n)
        
        loss, dW, dv = self.layer.loss_and_grads(X, y, M)
        
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(dW)))
        self.assertTrue(np.all(np.isfinite(dv)))

    def test_loss_and_grads_respects_mask(self):
        """Test that gradients respect the mask structure."""
        X = np.random.randn(20, self.n)
        y = np.random.randint(0, 2, 20).astype(float)
        
        # Create mask with some zero entries
        M = np.ones((self.n, self.n))
        M[0, 1] = 0.0
        M[2, 3] = 0.0
        
        _, dW, _ = self.layer.loss_and_grads(X, y, M)
        
        # Gradient should be zero where mask is zero
        self.assertEqual(dW[0, 1], 0.0)
        self.assertEqual(dW[2, 3], 0.0)

    def test_step_updates_parameters(self):
        """Test that step method updates parameters."""
        W_before = self.layer.W.copy()
        v_before = self.layer.v.copy()
        
        dW = np.random.randn(self.n, self.n)
        dv = np.random.randn(self.n)
        lr = 0.1
        
        self.layer.step(dW, dv, lr=lr)
        
        expected_W = W_before - lr * dW
        expected_v = v_before - lr * dv
        
        np.testing.assert_array_almost_equal(self.layer.W, expected_W)
        np.testing.assert_array_almost_equal(self.layer.v, expected_v)


class TestTrainForAlpha(unittest.TestCase):
    """Test suite for train_for_alpha function."""

    def test_train_for_alpha_returns_valid_accuracy(self):
        """Test that train_for_alpha returns a valid accuracy value."""
        accuracy = train_for_alpha(alpha=0.5, n=8, epochs=10, seed=42)
        
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_train_for_alpha_alpha_zero(self):
        """Test training with alpha=0 (pure topo0)."""
        accuracy = train_for_alpha(alpha=0.0, n=8, epochs=50, seed=42,
                                   topo0="chain", topo1="complete")
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_train_for_alpha_alpha_one(self):
        """Test training with alpha=1 (pure topo1)."""
        accuracy = train_for_alpha(alpha=1.0, n=8, epochs=50, seed=42,
                                   topo0="chain", topo1="complete")
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_train_for_alpha_different_topologies(self):
        """Test training with different topology combinations."""
        accuracy1 = train_for_alpha(alpha=0.5, n=8, epochs=30, seed=42,
                                    topo0="ring", topo1="star")
        accuracy2 = train_for_alpha(alpha=0.5, n=8, epochs=30, seed=42,
                                    topo0="chain", topo1="complete")
        
        # Both should return valid accuracies
        self.assertGreaterEqual(accuracy1, 0.0)
        self.assertLessEqual(accuracy1, 1.0)
        self.assertGreaterEqual(accuracy2, 0.0)
        self.assertLessEqual(accuracy2, 1.0)

    def test_train_for_alpha_deterministic(self):
        """Test that same seed produces same results."""
        accuracy1 = train_for_alpha(alpha=0.3, n=6, epochs=20, seed=100)
        accuracy2 = train_for_alpha(alpha=0.3, n=6, epochs=20, seed=100)
        
        self.assertAlmostEqual(accuracy1, accuracy2, places=10)

    def test_train_for_alpha_different_seeds(self):
        """Test that different seeds can produce different results."""
        accuracy1 = train_for_alpha(alpha=0.5, n=8, epochs=20, seed=1)
        accuracy2 = train_for_alpha(alpha=0.5, n=8, epochs=20, seed=999)
        
        # Results might differ due to different initializations
        # Both should still be valid
        self.assertGreaterEqual(accuracy1, 0.0)
        self.assertGreaterEqual(accuracy2, 0.0)

    def test_train_for_alpha_improves_over_baseline(self):
        """Test that training improves over random baseline."""
        # With enough epochs, accuracy should be better than random (0.5)
        accuracy = train_for_alpha(alpha=0.5, n=10, epochs=200, lr=0.2, seed=42)
        self.assertGreater(accuracy, 0.55)  # Should be better than random

    def test_train_for_alpha_with_different_params(self):
        """Test training with different hyperparameters."""
        acc_high_lr = train_for_alpha(alpha=0.5, n=8, epochs=50, lr=0.5, seed=42)
        acc_low_lr = train_for_alpha(alpha=0.5, n=8, epochs=50, lr=0.01, seed=42)
        
        # Both should return valid accuracies
        self.assertGreaterEqual(acc_high_lr, 0.0)
        self.assertLessEqual(acc_high_lr, 1.0)
        self.assertGreaterEqual(acc_low_lr, 0.0)
        self.assertLessEqual(acc_low_lr, 1.0)


class TestSweepAlphas(unittest.TestCase):
    """Test suite for sweep_alphas function."""

    def test_sweep_alphas_returns_correct_shape(self):
        """Test that sweep_alphas returns array with correct length."""
        alphas = np.array([0.0, 0.5, 1.0])
        results = sweep_alphas(alphas, n=6, seed=42)
        
        self.assertEqual(len(results), len(alphas))
        self.assertIsInstance(results, np.ndarray)

    def test_sweep_alphas_all_values_valid(self):
        """Test that all returned accuracies are valid."""
        alphas = np.linspace(0.0, 1.0, 5)
        results = sweep_alphas(alphas, n=8, seed=42)
        
        self.assertTrue(np.all(results >= 0.0))
        self.assertTrue(np.all(results <= 1.0))

    def test_sweep_alphas_single_alpha(self):
        """Test sweep with single alpha value."""
        alphas = np.array([0.5])
        results = sweep_alphas(alphas, n=6, seed=42)
        
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0], 0.0)
        self.assertLessEqual(results[0], 1.0)

    def test_sweep_alphas_deterministic(self):
        """Test that sweep with same seed produces same results."""
        alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        results1 = sweep_alphas(alphas, n=6, seed=42)
        results2 = sweep_alphas(alphas, n=6, seed=42)
        
        np.testing.assert_array_almost_equal(results1, results2)

    def test_sweep_alphas_different_topologies(self):
        """Test sweep with different topology pairs."""
        alphas = np.array([0.0, 0.5, 1.0])
        results = sweep_alphas(alphas, n=8, seed=42, 
                              topo0="ring", topo1="star")
        
        self.assertEqual(len(results), len(alphas))
        self.assertTrue(np.all(results >= 0.0))
        self.assertTrue(np.all(results <= 1.0))

    def test_sweep_alphas_multiple_values(self):
        """Test sweep orchestrates multiple training runs correctly."""
        alphas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        results = sweep_alphas(alphas, n=8, seed=42)
        
        # Should have one result per alpha
        self.assertEqual(len(results), 6)
        
        # All results should be valid accuracies
        for acc in results:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
            self.assertIsInstance(acc, (float, np.floating))

    def test_sweep_alphas_aggregates_correctly(self):
        """Test that results are aggregated in correct order."""
        alphas = np.array([0.0, 1.0])
        results = sweep_alphas(alphas, n=6, seed=42)
        
        # Manually compute expected results
        expected_0 = train_for_alpha(0.0, n=6, seed=42)
        expected_1 = train_for_alpha(1.0, n=6, seed=42)
        
        np.testing.assert_almost_equal(results[0], expected_0)
        np.testing.assert_almost_equal(results[1], expected_1)

    def test_sweep_alphas_empty_array(self):
        """Test sweep with empty alpha array."""
        alphas = np.array([])
        results = sweep_alphas(alphas, n=6, seed=42)
        
        self.assertEqual(len(results), 0)


class TestSigmoid(unittest.TestCase):
    """Test suite for sigmoid helper function."""

    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        result = sigmoid(np.array([0.0]))
        np.testing.assert_almost_equal(result, [0.5])

    def test_sigmoid_large_positive(self):
        """Test sigmoid saturates for large positive values."""
        result = sigmoid(np.array([100.0]))
        np.testing.assert_almost_equal(result, [1.0])

    def test_sigmoid_large_negative(self):
        """Test sigmoid saturates for large negative values."""
        result = sigmoid(np.array([-100.0]))
        np.testing.assert_almost_equal(result, [0.0], decimal=5)

    def test_sigmoid_range(self):
        """Test sigmoid output is always in (0, 1)."""
        x = np.linspace(-10, 10, 100)
        result = sigmoid(x)
        self.assertTrue(np.all(result > 0.0))
        self.assertTrue(np.all(result < 1.0))


class TestMakeSynthetic(unittest.TestCase):
    """Test suite for make_synthetic helper function."""

    def test_make_synthetic_shapes(self):
        """Test that make_synthetic returns correct shapes."""
        from sqnt_hardware_demo.train_demo import make_synthetic
        
        n = 10
        batch = 256
        X, y = make_synthetic(n=n, batch=batch, seed=42)
        
        self.assertEqual(X.shape, (batch, n))
        self.assertEqual(y.shape, (batch,))

    def test_make_synthetic_deterministic(self):
        """Test that same seed produces same data."""
        from sqnt_hardware_demo.train_demo import make_synthetic
        
        X1, y1 = make_synthetic(n=8, batch=100, seed=123)
        X2, y2 = make_synthetic(n=8, batch=100, seed=123)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_make_synthetic_labels_binary(self):
        """Test that labels are binary (0 or 1)."""
        from sqnt_hardware_demo.train_demo import make_synthetic
        
        _, y = make_synthetic(n=8, batch=200, seed=42)
        self.assertTrue(np.all((y == 0.0) | (y == 1.0)))


if __name__ == '__main__':
    unittest.main()
