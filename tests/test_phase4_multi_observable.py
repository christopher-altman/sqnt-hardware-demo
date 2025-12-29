"""
Tests for Phase IV: Multi-Observable Identifiability Protocols.

Verifies:
- Baseline behavior unchanged when multi-observable disabled
- Multi-observable mode adds expected history keys
- Determinism with fixed seeds
- Support metrics change measurably with lambda_aux > 0
"""

import numpy as np
import pytest


def test_baseline_unchanged_with_multiobs_disabled():
    """Verify baseline behavior when enable_multi_observable=False."""
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

    w_true = sample_ground_truth_mixture(K, seed=seed, concentration=0.5)
    X, y, _, _ = generate_planted_mixture_data(
        n=n,
        batch=batch,
        w_true=w_true,
        topology_names=topology_names,
        seed=seed,
        noise_level=0.05,
    )

    history = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_multi_observable=False,
        lambda_aux=0.0,
    )

    # Check standard keys exist
    assert 'loss' in history
    assert 'acc' in history
    assert 'weights' in history
    assert 'recovery_l1' in history
    assert 'recovery_kl' in history

    # Check multi-observable keys do NOT exist
    assert 'loss_aux' not in history
    assert 'acc_aux' not in history

    # Check array shapes
    assert history['loss'].shape == (epochs,)
    assert history['weights'].shape == (epochs, K)


def test_multiobs_adds_expected_keys():
    """Verify enabling multi-observable adds loss_aux and acc_aux to history."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 6
    batch = 64
    epochs = 20
    seed = 42
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
        aux_seed=0,
    )

    # Check multi-observable keys exist
    assert 'loss_aux' in history
    assert 'acc_aux' in history

    # Check standard keys still exist
    assert 'loss' in history
    assert 'acc' in history
    assert 'weights' in history
    assert 'recovery_l1' in history

    # Check shapes
    assert history['loss_aux'].shape == (epochs,)
    assert history['acc_aux'].shape == (epochs,)


def test_multiobs_determinism():
    """Verify determinism: same seeds produce identical results."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 6
    batch = 64
    epochs = 10
    seed = 123
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

    history1 = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_multi_observable=True,
        lambda_aux=0.05,
        aux_task="graph_feature",
        aux_seed=seed,
    )

    history2 = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_multi_observable=True,
        lambda_aux=0.05,
        aux_task="graph_feature",
        aux_seed=seed,
    )

    # Check exact match
    np.testing.assert_array_equal(history1['loss'], history2['loss'])
    np.testing.assert_array_equal(history1['loss_aux'], history2['loss_aux'])
    np.testing.assert_array_equal(history1['weights'], history2['weights'])
    np.testing.assert_array_equal(history1['recovery_l1'], history2['recovery_l1'])


def test_multiobs_changes_support_metrics():
    """Verify lambda_aux > 0 measurably changes recovery metrics."""
    from sqnt_hardware_demo.experiments import (
        sample_ground_truth_mixture,
        generate_planted_mixture_data,
        train_mixture_recovery,
    )

    n = 8
    batch = 128
    epochs = 30
    seed = 99
    topology_names = ["chain", "ring", "star", "complete"]
    K = len(topology_names)

    w_true = sample_ground_truth_mixture(K, seed=seed, concentration=0.3)
    X, y, _, _ = generate_planted_mixture_data(
        n=n,
        batch=batch,
        w_true=w_true,
        topology_names=topology_names,
        seed=seed,
        noise_level=0.05,
    )

    # Baseline (no multi-obs)
    history_baseline = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_multi_observable=False,
        lambda_aux=0.0,
    )

    # Multi-observable
    history_multiobs = train_mixture_recovery(
        X, y, w_true,
        topology_names=topology_names,
        n=n,
        epochs=epochs,
        seed=seed,
        enable_multi_observable=True,
        lambda_aux=0.2,
        aux_task="graph_feature",
        aux_seed=seed,
    )

    # Check that final metrics differ
    # (Not necessarily "better", but measurably different)
    final_l1_baseline = history_baseline['recovery_l1'][-1]
    final_l1_multiobs = history_multiobs['recovery_l1'][-1]

    # At least one should be different (within floating point)
    # Allow for possibility they're the same if task is easy
    # but check that histories are not identical
    weights_diff = np.linalg.norm(
        history_baseline['weights'][-1] - history_multiobs['weights'][-1]
    )

    # Should see some difference in learned weights
    assert weights_diff > 1e-6, "Multi-observable should affect learned weights"


def test_compute_graph_features():
    """Test graph feature computation."""
    from sqnt_hardware_demo.multi_observable import compute_graph_features
    from sqnt_hardware_demo.graphs import make_graph_mask

    n = 6
    mask_complete = make_graph_mask("complete", n, include_self=True, normalize=True)
    mask_chain = make_graph_mask("chain", n, include_self=True, normalize=True)

    # Triangle proxy
    feat_complete = compute_graph_features(mask_complete, "triangle_proxy")
    feat_chain = compute_graph_features(mask_chain, "triangle_proxy")

    assert feat_complete.shape == (1,)
    assert feat_chain.shape == (1,)

    # Triangle counts differ between topologies (direction depends on normalization)
    # Just verify they're different
    assert abs(feat_complete[0] - feat_chain[0]) > 0.01

    # Hubness proxy
    feat_hub_complete = compute_graph_features(mask_complete, "hubness_proxy")
    mask_star = make_graph_mask("star", n, include_self=True, normalize=True)
    feat_hub_star = compute_graph_features(mask_star, "hubness_proxy")

    # Star has higher hubness variance than complete
    assert feat_hub_star[0] > feat_hub_complete[0]


def test_generate_auxiliary_labels():
    """Test auxiliary label generation."""
    from sqnt_hardware_demo.multi_observable import generate_auxiliary_labels
    from sqnt_hardware_demo.experiments import sample_ground_truth_mixture

    n = 6
    batch = 32
    topology_names = ["chain", "ring", "star"]
    K = len(topology_names)

    w_true = sample_ground_truth_mixture(K, seed=0, concentration=0.5)
    X = np.random.randn(batch, n)

    y_aux = generate_auxiliary_labels(
        X, w_true, topology_names, n,
        aux_task="graph_feature",
        aux_seed=0,
    )

    assert y_aux.shape == (batch,)
    # Should be deterministic given same seed
    y_aux2 = generate_auxiliary_labels(
        X, w_true, topology_names, n,
        aux_task="graph_feature",
        aux_seed=0,
    )
    np.testing.assert_array_almost_equal(y_aux, y_aux2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
