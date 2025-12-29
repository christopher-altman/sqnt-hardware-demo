"""
Tests for Phase V: Adaptive Topology Learning (AQN).

Verifies:
- Baseline behavior unchanged when adaptive disabled
- Adaptive mode adds expected history keys
- Determinism with fixed seeds
- Adaptive updates actually change logits
- Weights remain on simplex
"""

import numpy as np
import pytest


def test_baseline_unchanged_with_adaptive_disabled():
    """Verify baseline behavior when enable_adaptive_topology=False."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 6
    batch = 64
    epochs = 20
    seed = 42
    topology_names = ["chain", "ring", "star"]
    K = len(topology_names)

    w_true = sample_ground_truth_mixture(K, seed=seed)
    X, y, _, _ = generate_planted_mixture_data(
        n=n,
        batch=batch,
        w_true=w_true,
        topology_names=topology_names,
        seed=seed,
    )

    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_adaptive_topology=False,
        adaptive_beta=0.0,
    )

    # Check standard keys exist
    assert 'loss' in history
    assert 'acc' in history
    assert 'weights' in history
    assert 'recovery_l1' in history

    # Check adaptive keys do NOT exist
    assert 'z_logits' not in history
    assert 'adaptive_step_norm' not in history


def test_adaptive_adds_expected_keys():
    """Verify enabling adaptive topology adds z_logits and adaptive_step_norm."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 6
    batch = 64
    epochs = 30
    seed = 42
    topology_names = ["chain", "complete"]
    K = len(topology_names)

    w_true = sample_ground_truth_mixture(K, seed=seed)
    X, y, _, _ = generate_planted_mixture_data(
        n=n,
        batch=batch,
        w_true=w_true,
        topology_names=topology_names,
        seed=seed,
    )

    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_adaptive_topology=True,
        adaptive_beta=0.2,
        adaptive_momentum=0.5,
        adaptive_update="momentum",
    )

    # Check adaptive keys exist
    assert 'z_logits' in history
    assert 'adaptive_step_norm' in history

    # Check standard keys still exist
    assert 'loss' in history
    assert 'acc' in history
    assert 'weights' in history

    # Check shapes
    assert history['adaptive_step_norm'].shape == (epochs,)
    # z_logits sampled every 10 epochs
    expected_z_snapshots = epochs // 10
    assert len(history['z_logits']) == expected_z_snapshots


def test_adaptive_determinism():
    """Verify determinism: same seeds produce identical results."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 6
    batch = 64
    epochs = 15
    seed = 777
    topology_names = ["ring", "star"]
    K = len(topology_names)

    w_true = sample_ground_truth_mixture(K, seed=seed)
    X, y, _, _ = generate_planted_mixture_data(
        n=n,
        batch=batch,
        w_true=w_true,
        topology_names=topology_names,
        seed=seed,
    )

    history1 = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_adaptive_topology=True,
        adaptive_beta=0.1,
        adaptive_momentum=0.3,
        adaptive_update="momentum",
    )

    history2 = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_adaptive_topology=True,
        adaptive_beta=0.1,
        adaptive_momentum=0.3,
        adaptive_update="momentum",
    )

    # Check exact match
    np.testing.assert_array_equal(history1['loss'], history2['loss'])
    np.testing.assert_array_equal(history1['weights'], history2['weights'])
    np.testing.assert_array_equal(history1['adaptive_step_norm'], history2['adaptive_step_norm'])
    np.testing.assert_array_equal(history1['z_logits'], history2['z_logits'])


def test_adaptive_changes_logits():
    """Verify adaptive updates actually change logits relative to static baseline."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 6
    batch = 128
    epochs = 40
    seed = 999
    topology_names = ["chain", "ring", "star", "complete"]
    K = len(topology_names)

    w_true = sample_ground_truth_mixture(K, seed=seed, concentration=0.5)
    X, y, _, _ = generate_planted_mixture_data(
        n=n,
        batch=batch,
        w_true=w_true,
        topology_names=topology_names,
        seed=seed,
    )

    # Baseline (no adaptive)
    history_baseline = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_adaptive_topology=False,
    )

    # Adaptive
    history_adaptive = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_adaptive_topology=True,
        adaptive_beta=0.3,
        adaptive_momentum=0.4,
        adaptive_update="momentum",
    )

    # Check that adaptive step norms are present and nonzero
    assert history_adaptive['adaptive_step_norm'].mean() > 0

    # Check that final weights differ
    final_weights_baseline = history_baseline['weights'][-1]
    final_weights_adaptive = history_adaptive['weights'][-1]

    weights_diff = np.linalg.norm(final_weights_baseline - final_weights_adaptive)
    assert weights_diff > 1e-6, "Adaptive updates should change final weights"


def test_weights_remain_on_simplex():
    """Verify weights stay on simplex: sum=1, all >= 0."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 8
    batch = 64
    epochs = 25
    seed = 111
    topology_names = ["chain", "ring", "complete"]
    K = len(topology_names)

    w_true = sample_ground_truth_mixture(K, seed=seed)
    X, y, _, _ = generate_planted_mixture_data(
        n=n,
        batch=batch,
        w_true=w_true,
        topology_names=topology_names,
        seed=seed,
    )

    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_adaptive_topology=True,
        adaptive_beta=0.2,
        adaptive_momentum=0.5,
        adaptive_update="ema",
    )

    # Check all weight vectors
    for epoch_weights in history['weights']:
        # Sum should be 1
        np.testing.assert_almost_equal(epoch_weights.sum(), 1.0, decimal=5)
        # All should be >= 0
        assert np.all(epoch_weights >= 0), f"Negative weight found: {epoch_weights}"


def test_ema_update_mode():
    """Test EMA update mode."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 6
    batch = 64
    epochs = 20
    seed = 555
    topology_names = ["star", "complete"]
    K = len(topology_names)

    w_true = sample_ground_truth_mixture(K, seed=seed)
    X, y, _, _ = generate_planted_mixture_data(
        n=n,
        batch=batch,
        w_true=w_true,
        topology_names=topology_names,
        seed=seed,
    )

    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_adaptive_topology=True,
        adaptive_beta=0.15,
        adaptive_update="ema",
    )

    # Should complete without error
    assert 'adaptive_step_norm' in history
    assert history['adaptive_step_norm'].mean() > 0


def test_adaptive_with_multi_observable():
    """Test combining adaptive topology with multi-observable."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 6
    batch = 64
    epochs = 20
    seed = 333
    topology_names = ["chain", "ring"]
    K = len(topology_names)

    w_true = sample_ground_truth_mixture(K, seed=seed)
    X, y, _, _ = generate_planted_mixture_data(
        n=n,
        batch=batch,
        w_true=w_true,
        topology_names=topology_names,
        seed=seed,
    )

    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_multi_observable=True,
        lambda_aux=0.1,
        aux_task="graph_feature",
        enable_adaptive_topology=True,
        adaptive_beta=0.2,
        adaptive_momentum=0.3,
    )

    # Should have keys from both Phase IV and Phase V
    assert 'loss_aux' in history
    assert 'acc_aux' in history
    assert 'adaptive_step_norm' in history
    assert 'z_logits' in history

    # Standard keys
    assert 'loss' in history
    assert 'acc' in history
    assert 'weights' in history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
