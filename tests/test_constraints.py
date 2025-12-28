"""
Test suite for hardware constraint operators.

Tests cover:
1. violates_max_degree with different topologies and max_degree values
2. violates_locality with different topologies and locality_radius values
3. constraint_penalty correctly calculates penalty for violating topologies
4. constraint_penalty_grad correctly calculates gradients for violating topologies
5. train_mixture_recovery applies constraint penalty to loss and gradients when constraints are enabled
"""

import unittest
import numpy as np
from sqnt_hardware_demo.constraints import (
    violates_max_degree,
    violates_locality,
    constraint_penalty,
    constraint_penalty_grad,
)
from sqnt_hardware_demo.experiments import (
    sample_ground_truth_mixture,
    generate_planted_mixture_data,
    train_mixture_recovery,
)


class TestViolatesMaxDegree(unittest.TestCase):
    """Test suite for violates_max_degree function."""

    def test_chain_low_degree(self):
        """Test that chain topology does not violate low max_degree."""
        # Chain has max degree 2
        self.assertFalse(violates_max_degree("chain", max_degree=2))
        self.assertFalse(violates_max_degree("chain", max_degree=3))
        self.assertFalse(violates_max_degree("chain", max_degree=10))

    def test_chain_too_low_degree(self):
        """Test that chain violates very low max_degree."""
        # Chain has max degree 2, so max_degree=1 should be violated
        self.assertTrue(violates_max_degree("chain", max_degree=1))

    def test_ring_low_degree(self):
        """Test that ring topology does not violate low max_degree."""
        # Ring has max degree 2
        self.assertFalse(violates_max_degree("ring", max_degree=2))
        self.assertFalse(violates_max_degree("ring", max_degree=5))

    def test_ring_too_low_degree(self):
        """Test that ring violates very low max_degree."""
        self.assertTrue(violates_max_degree("ring", max_degree=1))

    def test_star_high_degree(self):
        """Test that star topology violates low max_degree."""
        # Star has hub with degree n-1 (high degree)
        self.assertTrue(violates_max_degree("star", max_degree=2))
        self.assertTrue(violates_max_degree("star", max_degree=5))
        self.assertTrue(violates_max_degree("star", max_degree=10))

    def test_complete_high_degree(self):
        """Test that complete topology violates low max_degree."""
        # Complete has degree n-1 for all nodes (high degree)
        self.assertTrue(violates_max_degree("complete", max_degree=2))
        self.assertTrue(violates_max_degree("complete", max_degree=5))
        self.assertTrue(violates_max_degree("complete", max_degree=10))

    def test_case_insensitive(self):
        """Test that topology names are case insensitive."""
        self.assertFalse(violates_max_degree("CHAIN", max_degree=2))
        self.assertFalse(violates_max_degree("Chain", max_degree=2))
        self.assertFalse(violates_max_degree("  chain  ", max_degree=2))

    def test_unknown_topology(self):
        """Test that unknown topology returns False (conservative)."""
        self.assertFalse(violates_max_degree("unknown", max_degree=2))
        self.assertFalse(violates_max_degree("", max_degree=2))

    def test_various_max_degrees(self):
        """Test different max_degree thresholds."""
        # Chain should only violate max_degree < 2
        for degree in range(1, 10):
            if degree < 2:
                self.assertTrue(violates_max_degree("chain", max_degree=degree))
            else:
                self.assertFalse(violates_max_degree("chain", max_degree=degree))


class TestViolatesLocality(unittest.TestCase):
    """Test suite for violates_locality function."""

    def test_chain_local(self):
        """Test that chain topology is always local."""
        # Chain has radius 1 (neighbors only)
        self.assertFalse(violates_locality("chain", locality_radius=1))
        self.assertFalse(violates_locality("chain", locality_radius=2))
        self.assertFalse(violates_locality("chain", locality_radius=10))

    def test_ring_local(self):
        """Test that ring topology is always local."""
        # Ring has radius 1 (neighbors only)
        self.assertFalse(violates_locality("ring", locality_radius=1))
        self.assertFalse(violates_locality("ring", locality_radius=2))
        self.assertFalse(violates_locality("ring", locality_radius=10))

    def test_star_radius_2(self):
        """Test that star topology has effective radius 2."""
        # Star has radius 2 (spoke-to-spoke via hub)
        self.assertTrue(violates_locality("star", locality_radius=1))
        self.assertFalse(violates_locality("star", locality_radius=2))
        self.assertFalse(violates_locality("star", locality_radius=3))

    def test_complete_non_local(self):
        """Test that complete topology violates small locality radius."""
        # Complete graph has all pairs, violates small radius
        self.assertTrue(violates_locality("complete", locality_radius=1))
        self.assertTrue(violates_locality("complete", locality_radius=2))
        self.assertTrue(violates_locality("complete", locality_radius=10))

    def test_case_insensitive(self):
        """Test that topology names are case insensitive."""
        self.assertFalse(violates_locality("CHAIN", locality_radius=1))
        self.assertFalse(violates_locality("Ring", locality_radius=1))
        self.assertTrue(violates_locality("  STAR  ", locality_radius=1))

    def test_unknown_topology(self):
        """Test that unknown topology returns False (conservative)."""
        self.assertFalse(violates_locality("unknown", locality_radius=1))
        self.assertFalse(violates_locality("", locality_radius=1))


class TestConstraintPenalty(unittest.TestCase):
    """Test suite for constraint_penalty function."""

    def test_disabled_constraints(self):
        """Test that disabled constraints return zero penalty."""
        weights = np.array([0.5, 0.5])
        topologies = ["chain", "complete"]
        
        # No config
        penalty = constraint_penalty(weights, topologies, cfg=None)
        self.assertEqual(penalty, 0.0)
        
        # Disabled
        cfg = {"enabled": False, "max_degree": 2}
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertEqual(penalty, 0.0)

    def test_no_constraints_specified(self):
        """Test that enabled but empty constraints return zero penalty."""
        weights = np.array([0.5, 0.5])
        topologies = ["chain", "complete"]
        cfg = {"enabled": True}
        
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertEqual(penalty, 0.0)

    def test_max_degree_constraint_no_violations(self):
        """Test penalty when no topologies violate max_degree."""
        weights = np.array([0.3, 0.7])
        topologies = ["chain", "ring"]
        cfg = {"enabled": True, "max_degree": 2}
        
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertEqual(penalty, 0.0)

    def test_max_degree_constraint_with_violations(self):
        """Test penalty when some topologies violate max_degree."""
        weights = np.array([0.3, 0.7])
        topologies = ["chain", "complete"]
        cfg = {"enabled": True, "max_degree": 2}
        
        # Only complete violates (has high degree)
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertAlmostEqual(penalty, 0.7, places=10)

    def test_locality_constraint_no_violations(self):
        """Test penalty when no topologies violate locality."""
        weights = np.array([0.4, 0.6])
        topologies = ["chain", "ring"]
        cfg = {"enabled": True, "locality_radius": 1}
        
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertEqual(penalty, 0.0)

    def test_locality_constraint_with_violations(self):
        """Test penalty when some topologies violate locality."""
        weights = np.array([0.2, 0.8])
        topologies = ["chain", "complete"]
        cfg = {"enabled": True, "locality_radius": 1}
        
        # Only complete violates (non-local)
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertAlmostEqual(penalty, 0.8, places=10)

    def test_combined_constraints(self):
        """Test penalty with both max_degree and locality constraints."""
        weights = np.array([0.2, 0.3, 0.5])
        topologies = ["chain", "star", "complete"]
        cfg = {"enabled": True, "max_degree": 2, "locality_radius": 1}
        
        # Both star and complete violate (star: degree, complete: both)
        # Penalty should be 0.3 + 0.5 = 0.8
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertAlmostEqual(penalty, 0.8, places=10)

    def test_all_topologies_violate(self):
        """Test penalty when all topologies violate constraints."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        topologies = ["star", "star", "complete", "complete"]
        cfg = {"enabled": True, "max_degree": 2}
        
        # All violate
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertAlmostEqual(penalty, 1.0, places=10)

    def test_no_topologies_violate(self):
        """Test penalty when no topologies violate constraints."""
        weights = np.array([0.3, 0.3, 0.4])
        topologies = ["chain", "ring", "chain"]
        cfg = {"enabled": True, "max_degree": 2, "locality_radius": 1}
        
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertEqual(penalty, 0.0)

    def test_zero_weight_on_violating_topology(self):
        """Test that zero-weight violations don't contribute to penalty."""
        weights = np.array([1.0, 0.0])
        topologies = ["chain", "complete"]
        cfg = {"enabled": True, "max_degree": 2}
        
        # Complete violates but has zero weight
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertEqual(penalty, 0.0)

    def test_penalty_returns_float(self):
        """Test that penalty always returns a float."""
        weights = np.array([0.5, 0.5])
        topologies = ["chain", "complete"]
        cfg = {"enabled": True, "max_degree": 2}
        
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        self.assertIsInstance(penalty, float)


class TestConstraintPenaltyGrad(unittest.TestCase):
    """Test suite for constraint_penalty_grad function."""

    def test_disabled_constraints(self):
        """Test that disabled constraints return zero gradient."""
        weights = np.array([0.5, 0.5])
        topologies = ["chain", "complete"]
        
        # No config
        grad = constraint_penalty_grad(weights, topologies, cfg=None)
        np.testing.assert_array_equal(grad, np.zeros(2))
        
        # Disabled
        cfg = {"enabled": False, "max_degree": 2}
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        np.testing.assert_array_equal(grad, np.zeros(2))

    def test_no_constraints_specified(self):
        """Test that enabled but empty constraints return zero gradient."""
        weights = np.array([0.5, 0.5])
        topologies = ["chain", "complete"]
        cfg = {"enabled": True}
        
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        np.testing.assert_array_equal(grad, np.zeros(2))

    def test_max_degree_gradient_no_violations(self):
        """Test gradient when no topologies violate max_degree."""
        weights = np.array([0.3, 0.7])
        topologies = ["chain", "ring"]
        cfg = {"enabled": True, "max_degree": 2}
        
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        np.testing.assert_array_equal(grad, np.zeros(2))

    def test_max_degree_gradient_with_violations(self):
        """Test gradient when some topologies violate max_degree."""
        weights = np.array([0.3, 0.7])
        topologies = ["chain", "complete"]
        cfg = {"enabled": True, "max_degree": 2}
        
        # Only complete violates
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        expected = np.array([0.0, 1.0])
        np.testing.assert_array_equal(grad, expected)

    def test_locality_gradient_with_violations(self):
        """Test gradient when some topologies violate locality."""
        weights = np.array([0.2, 0.8])
        topologies = ["chain", "complete"]
        cfg = {"enabled": True, "locality_radius": 1}
        
        # Only complete violates
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        expected = np.array([0.0, 1.0])
        np.testing.assert_array_equal(grad, expected)

    def test_combined_constraints_gradient(self):
        """Test gradient with both max_degree and locality constraints."""
        weights = np.array([0.2, 0.3, 0.5])
        topologies = ["chain", "star", "complete"]
        cfg = {"enabled": True, "max_degree": 2, "locality_radius": 1}
        
        # Both star and complete violate
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        expected = np.array([0.0, 1.0, 1.0])
        np.testing.assert_array_equal(grad, expected)

    def test_all_topologies_violate_gradient(self):
        """Test gradient when all topologies violate constraints."""
        weights = np.array([0.25, 0.25, 0.5])
        topologies = ["star", "complete", "complete"]
        cfg = {"enabled": True, "max_degree": 2}
        
        # All violate
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(grad, expected)

    def test_gradient_shape(self):
        """Test that gradient has same shape as weights."""
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        topologies = ["chain", "ring", "star", "complete"]
        cfg = {"enabled": True, "max_degree": 2}
        
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        self.assertEqual(grad.shape, weights.shape)

    def test_gradient_binary_values(self):
        """Test that gradient values are 0 or 1."""
        weights = np.array([0.2, 0.3, 0.5])
        topologies = ["chain", "star", "complete"]
        cfg = {"enabled": True, "max_degree": 2, "locality_radius": 1}
        
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        # All values should be 0 or 1
        self.assertTrue(np.all((grad == 0.0) | (grad == 1.0)))

    def test_gradient_consistency_with_penalty(self):
        """Test that gradient is consistent with penalty function."""
        weights = np.array([0.2, 0.3, 0.5])
        topologies = ["chain", "star", "complete"]
        cfg = {"enabled": True, "max_degree": 2}
        
        penalty = constraint_penalty(weights, topologies, cfg=cfg)
        grad = constraint_penalty_grad(weights, topologies, cfg=cfg)
        
        # Penalty should equal sum of weights where grad is 1
        expected_penalty = np.sum(weights * grad)
        self.assertAlmostEqual(penalty, expected_penalty, places=10)


class TestTrainMixtureRecoveryWithConstraints(unittest.TestCase):
    """Test suite for train_mixture_recovery with constraint penalties."""

    def test_constraints_disabled(self):
        """Test that training works with constraints disabled."""
        n = 8
        K = 2
        topologies = ["chain", "ring"]
        
        w_true = sample_ground_truth_mixture(K, seed=0)
        X, y, _, _ = generate_planted_mixture_data(
            n=n, batch=128, w_true=w_true,
            topology_names=topologies, seed=0
        )
        
        # No constraint config
        history = train_mixture_recovery(
            X, y, w_true,
            topology_names=topologies,
            n=n, epochs=10, seed=0
        )
        
        self.assertIn('loss', history)
        self.assertIn('weights', history)
        self.assertEqual(len(history['loss']), 10)

    def test_constraints_enabled_penalty_in_loss(self):
        """Test that constraint penalty affects loss when enabled."""
        n = 8
        K = 2
        topologies = ["chain", "complete"]  # complete violates constraints
        
        w_true = np.array([0.9, 0.1])  # True mixture favors chain
        X, y, _, _ = generate_planted_mixture_data(
            n=n, batch=128, w_true=w_true,
            topology_names=topologies, seed=0
        )
        
        # Train without constraints
        history_no_constraint = train_mixture_recovery(
            X, y, w_true,
            topology_names=topologies,
            n=n, epochs=50, lr_mixture=0.1, seed=42,
            constraint_cfg=None
        )
        
        # Train with constraints (penalize complete)
        constraint_cfg = {
            "enabled": True,
            "max_degree": 2,
            "lambda_constraint": 1.0
        }
        history_with_constraint = train_mixture_recovery(
            X, y, w_true,
            topology_names=topologies,
            n=n, epochs=50, lr_mixture=0.1, seed=42,
            constraint_cfg=constraint_cfg
        )
        
        # Final learned weights should be different
        final_weights_no = history_no_constraint['weights'][-1]
        final_weights_with = history_with_constraint['weights'][-1]
        
        # With constraints, weight on complete should be lower
        self.assertLess(
            final_weights_with[1],
            final_weights_no[1],
            msg="Constraint should reduce weight on complete topology"
        )

    def test_constraints_gradient_applied(self):
        """Test that constraint gradients affect learning."""
        n = 8
        K = 3
        topologies = ["chain", "star", "complete"]
        
        w_true = np.array([0.5, 0.3, 0.2])
        X, y, _, _ = generate_planted_mixture_data(
            n=n, batch=128, w_true=w_true,
            topology_names=topologies, seed=0
        )
        
        # Strong constraint penalty
        constraint_cfg = {
            "enabled": True,
            "max_degree": 2,
            "locality_radius": 1,
            "lambda_constraint": 5.0  # Strong penalty
        }
        
        history = train_mixture_recovery(
            X, y, w_true,
            topology_names=topologies,
            n=n, epochs=100, lr_mixture=0.2, seed=42,
            constraint_cfg=constraint_cfg
        )
        
        # Final weights: chain should have highest weight (satisfies both constraints)
        final_weights = history['weights'][-1]
        
        # Chain (index 0) should have higher weight than star and complete
        # since both star and complete violate constraints
        self.assertGreater(
            final_weights[0],
            final_weights[1],
            msg="Chain should have higher weight than star under constraints"
        )
        self.assertGreater(
            final_weights[0],
            final_weights[2],
            msg="Chain should have higher weight than complete under constraints"
        )

    def test_locality_constraint_effect(self):
        """Test that locality constraint specifically affects non-local topologies."""
        n = 8
        K = 2
        topologies = ["ring", "complete"]
        
        w_true = np.array([0.5, 0.5])
        X, y, _, _ = generate_planted_mixture_data(
            n=n, batch=256, w_true=w_true,
            topology_names=topologies, seed=0
        )
        
        # Locality constraint only
        constraint_cfg = {
            "enabled": True,
            "locality_radius": 1,
            "lambda_constraint": 2.0
        }
        
        history = train_mixture_recovery(
            X, y, w_true,
            topology_names=topologies,
            n=n, epochs=100, lr_mixture=0.2, seed=42,
            constraint_cfg=constraint_cfg
        )
        
        final_weights = history['weights'][-1]
        
        # Ring satisfies locality, complete does not
        # So ring should have higher final weight
        self.assertGreater(
            final_weights[0],
            final_weights[1],
            msg="Ring should have higher weight than complete under locality constraint"
        )

    def test_constraint_penalty_strength(self):
        """Test that stronger penalty leads to more constraint satisfaction."""
        n = 8
        K = 2
        topologies = ["chain", "complete"]
        
        w_true = np.array([0.3, 0.7])  # True favors complete
        X, y, _, _ = generate_planted_mixture_data(
            n=n, batch=256, w_true=w_true,
            topology_names=topologies, seed=0
        )
        
        # Weak penalty
        constraint_cfg_weak = {
            "enabled": True,
            "max_degree": 2,
            "lambda_constraint": 0.5
        }
        history_weak = train_mixture_recovery(
            X, y, w_true,
            topology_names=topologies,
            n=n, epochs=100, lr_mixture=0.2, seed=42,
            constraint_cfg=constraint_cfg_weak
        )
        
        # Strong penalty
        constraint_cfg_strong = {
            "enabled": True,
            "max_degree": 2,
            "lambda_constraint": 5.0
        }
        history_strong = train_mixture_recovery(
            X, y, w_true,
            topology_names=topologies,
            n=n, epochs=100, lr_mixture=0.2, seed=42,
            constraint_cfg=constraint_cfg_strong
        )
        
        # Stronger penalty should push more weight to chain
        final_chain_weight_weak = history_weak['weights'][-1, 0]
        final_chain_weight_strong = history_strong['weights'][-1, 0]
        
        self.assertGreater(
            final_chain_weight_strong,
            final_chain_weight_weak,
            msg="Stronger constraint penalty should increase weight on feasible topology"
        )

    def test_no_violation_no_penalty(self):
        """Test that constraints don't affect learning when no violations occur."""
        n = 8
        K = 2
        topologies = ["chain", "ring"]  # Both satisfy constraints
        
        w_true = sample_ground_truth_mixture(K, seed=0)
        X, y, _, _ = generate_planted_mixture_data(
            n=n, batch=128, w_true=w_true,
            topology_names=topologies, seed=0
        )
        
        # Train without constraints
        history_no_constraint = train_mixture_recovery(
            X, y, w_true,
            topology_names=topologies,
            n=n, epochs=50, seed=42
        )
        
        # Train with constraints (but both topologies satisfy them)
        constraint_cfg = {
            "enabled": True,
            "max_degree": 2,
            "locality_radius": 1,
            "lambda_constraint": 10.0  # Very strong, but shouldn't matter
        }
        history_with_constraint = train_mixture_recovery(
            X, y, w_true,
            topology_names=topologies,
            n=n, epochs=50, seed=42,
            constraint_cfg=constraint_cfg
        )
        
        # Results should be nearly identical since no topology violates
        final_weights_no = history_no_constraint['weights'][-1]
        final_weights_with = history_with_constraint['weights'][-1]
        
        np.testing.assert_array_almost_equal(
            final_weights_no,
            final_weights_with,
            decimal=6,
            err_msg="Constraints should not affect learning when all topologies are feasible"
        )


if __name__ == '__main__':
    unittest.main()
