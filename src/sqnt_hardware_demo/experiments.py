"""
Ground-truth mixture recovery experiments for SQNT.

Implements the "aha" experiment: plant a known topology mixture,
generate synthetic data, train to recover the mixture, and verify convergence.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from .graphs import make_graph_mask
from .sqnt_layer import SQNTLayer, sigmoid
from .mixture import TopologyMixture, softmax, compute_dL_dM


def sample_ground_truth_mixture(K: int, seed: int = 0, concentration: float = 1.0) -> np.ndarray:
    """
    Sample a ground-truth mixture weight vector.

    Parameters
    ----------
    K : int
        Number of topologies.
    seed : int
        Random seed.
    concentration : float
        Dirichlet concentration parameter. Higher = more uniform.

    Returns
    -------
    w_true : np.ndarray
        Ground-truth mixture weights summing to 1, shape (K,).
    """
    rng = np.random.default_rng(seed)
    # Sample from Dirichlet for interpretable mixture
    alpha = np.ones(K) * concentration
    w_true = rng.dirichlet(alpha)
    return w_true


def generate_planted_mixture_data(
    n: int,
    batch: int,
    w_true: np.ndarray,
    topology_names: List[str],
    seed: int = 0,
    noise_level: float = 0.05,
    include_self: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a planted topology mixture.

    The data generation process:
    1. Build true mixture mask M_true = sum_k w_true_k * M_k
    2. Sample random true weights W_true, v_true
    3. Compute W_eff_true = W_true * M_true
    4. Generate labels: y = sigmoid(X @ W_eff_true.T @ v_true) > 0.5
    5. Add label noise

    Parameters
    ----------
    n : int
        Number of nodes.
    batch : int
        Number of samples.
    w_true : np.ndarray
        Ground-truth mixture weights, shape (K,).
    topology_names : List[str]
        List of topology names.
    seed : int
        Random seed.
    noise_level : float
        Probability of flipping each label.
    include_self : bool
        Include self-loops in masks.
    normalize : bool
        Row-normalize masks.

    Returns
    -------
    X : np.ndarray
        Input data, shape (batch, n).
    y : np.ndarray
        Binary labels, shape (batch,).
    W_true : np.ndarray
        True weight matrix (for reference), shape (n, n).
    v_true : np.ndarray
        True output vector (for reference), shape (n,).
    """
    rng = np.random.default_rng(seed)
    K = len(topology_names)

    # Build topology masks
    masks = [make_graph_mask(name, n, include_self=include_self, normalize=normalize)
             for name in topology_names]

    # Compute true mixture mask
    M_true = np.zeros((n, n))
    for k in range(K):
        M_true += w_true[k] * masks[k]

    # Sample true parameters
    W_true = rng.standard_normal((n, n)) * 0.3
    v_true = rng.standard_normal(n) * 0.3

    # Generate input data
    X = rng.standard_normal((batch, n))

    # Compute effective weights and forward pass
    W_eff_true = W_true * M_true
    h = X @ W_eff_true.T
    logits = h @ v_true
    p = sigmoid(logits)

    # Generate labels with noise
    y = (p > 0.5).astype(float)
    flip_mask = rng.random(batch) < noise_level
    y[flip_mask] = 1.0 - y[flip_mask]

    return X, y, W_true, v_true


def train_mixture_recovery(
    X: np.ndarray,
    y: np.ndarray,
    w_true: np.ndarray,
    topology_names: List[str],
    n: int,
    epochs: int = 300,
    lr_params: float = 0.2,
    lr_mixture: float = 0.15,
    seed: int = 0,
    include_self: bool = True,
    normalize: bool = True,
    lambda_sparsity: float = 0.0,
    lambda_entropy: float = 0.0,
    lambda_dirichlet: float = 0.0,
    alpha_dirichlet: float = 0.3,
    enable_compile_constraints: bool = False,
    device_graph: Optional[dict] = None,
    lambda_compile: float = 0.0,
    enable_multi_observable: bool = False,
    lambda_aux: float = 0.0,
    aux_task: str = "graph_feature",
    aux_seed: int = 0,
    enable_adaptive_topology: bool = False,
    adaptive_beta: float = 0.0,
    adaptive_momentum: float = 0.0,
    adaptive_update: str = "momentum",
) -> Dict:
    """
    Train to recover a planted mixture and track convergence.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (batch, n).
    y : np.ndarray
        Binary labels, shape (batch,).
    w_true : np.ndarray
        Ground-truth mixture weights for tracking recovery.
    topology_names : List[str]
        List of topology names.
    n : int
        Number of nodes.
    epochs : int
        Number of training epochs.
    lr_params : float
        Learning rate for W and v.
    lr_mixture : float
        Learning rate for mixture logits.
    seed : int
        Random seed.
    include_self : bool
        Include self-loops in masks.
    normalize : bool
        Row-normalize masks.
    lambda_sparsity : float
        L1 sparsity penalty on mixture mask.
    lambda_entropy : float
        Entropy regularization on weights (positive encourages uniform,
        negative encourages concentration).
    lambda_dirichlet : float
        Dirichlet MAP prior strength. With alpha_dirichlet < 1,
        encourages sparse mixtures by pushing small weights toward 0.
    alpha_dirichlet : float
        Dirichlet concentration parameter. Values < 1 encourage sparsity.
        Default 0.3 is a reasonable sparse prior.
    enable_compile_constraints : bool
        If True and device_graph is provided, include compilation penalty.
    device_graph : dict, optional
        Device graph dictionary for compilation-aware constraints.
    lambda_compile : float
        Compilation penalty strength. Only used if enable_compile_constraints
        is True and device_graph is provided.
    enable_multi_observable : bool
        If True and lambda_aux > 0, enables multi-observable training.
        Default False (Phase IV opt-in).
    lambda_aux : float
        Auxiliary loss weight. Only used if enable_multi_observable is True.
        Default 0.0.
    aux_task : str
        Auxiliary task type. Default "graph_feature".
    aux_seed : int
        Random seed for auxiliary label generation. Default 0.
    enable_adaptive_topology : bool
        If True and adaptive_beta > 0, enables adaptive topology dynamics.
        Default False (Phase V opt-in).
    adaptive_beta : float
        Inertia/EMA parameter for adaptive updates. 0 disables. Default 0.0.
    adaptive_momentum : float
        Momentum parameter. Default 0.0.
    adaptive_update : str
        Update rule: "momentum" or "ema". Default "momentum".

    Returns
    -------
    history : dict
        Training history with keys:
        - 'loss': (epochs,) loss values
        - 'acc': (epochs,) accuracy values
        - 'weights': (epochs, K) learned weights over time
        - 'weights_true': (K,) ground-truth weights
        - 'recovery_l1': (epochs,) L1 distance to true weights
        - 'recovery_kl': (epochs,) KL divergence from true weights
        - 'loss_compile': (epochs,) compilation penalty (if enabled)
        - 'loss_aux': (epochs,) auxiliary loss (if multi-observable enabled)
        - 'acc_aux': (epochs,) auxiliary metric (if multi-observable enabled)
        - 'z_logits': logit snapshots (if adaptive enabled)
        - 'adaptive_step_norm': adaptive update norms (if adaptive enabled)
    """
    # Dispatch to adaptive trainer if either Phase IV or Phase V is enabled
    if enable_multi_observable or enable_adaptive_topology:
        from .adaptive_topology import train_adaptive_topology
        return train_adaptive_topology(
            X=X,
            y=y,
            w_true=w_true,
            topology_names=topology_names,
            n=n,
            epochs=epochs,
            lr_params=lr_params,
            lr_mixture=lr_mixture,
            seed=seed,
            include_self=include_self,
            normalize=normalize,
            lambda_sparsity=lambda_sparsity,
            lambda_entropy=lambda_entropy,
            lambda_dirichlet=lambda_dirichlet,
            alpha_dirichlet=alpha_dirichlet,
            enable_compile_constraints=enable_compile_constraints,
            device_graph=device_graph,
            lambda_compile=lambda_compile,
            enable_adaptive_topology=enable_adaptive_topology,
            adaptive_beta=adaptive_beta,
            adaptive_momentum=adaptive_momentum,
            adaptive_update=adaptive_update,
            enable_multi_observable=enable_multi_observable,
            lambda_aux=lambda_aux,
            aux_task=aux_task,
            aux_seed=aux_seed,
        )
    # Build topology masks
    masks = [make_graph_mask(name, n, include_self=include_self, normalize=normalize)
             for name in topology_names]
    K = len(masks)

    # Initialize model and mixture
    model = SQNTLayer(n=n, seed=seed)
    mixture = TopologyMixture(masks, seed=seed)

    # Check if compilation constraints are enabled
    use_compile = (
        enable_compile_constraints
        and device_graph is not None
        and lambda_compile > 0
    )

    # History tracking
    history = {
        'loss': [],
        'acc': [],
        'weights': [],
        'weights_true': w_true.copy(),
        'recovery_l1': [],
        'recovery_kl': [],
    }
    if use_compile:
        history['loss_compile'] = []

    for epoch in range(epochs):
        # Get current mixture mask
        M = mixture.mixture_mask()
        w = mixture.weights()

        # Forward pass and compute loss/gradients
        loss, dW, dv = model.loss_and_grads(X, y, M)

        # Compute dL/dM for mixture gradient
        W_eff = model.W * M
        p = model.forward(X, M)
        dlogits = (p - y) / X.shape[0]
        h = X @ W_eff.T
        dH = np.outer(dlogits, model.v)
        dWeff = dH.T @ X
        dL_dM = compute_dL_dM(model.W, dWeff)

        # Compute gradient for mixture logits
        dL_dz = mixture.grad_z(dL_dM)

        # Add regularization gradients
        if lambda_sparsity > 0:
            # Sparsity: penalize L1 norm of mask
            # dL/dM_ij += lambda * sign(M_ij) but M is always positive
            dL_dM_sparse = lambda_sparsity * np.ones_like(M)
            dL_dz += mixture.grad_z(dL_dM_sparse)

        if lambda_entropy != 0:
            # Entropy regularization: H(w) = -sum_k w_k log(w_k)
            # dH/dw_k = -(log(w_k) + 1)
            # For loss, we add -lambda * H (positive lambda encourages entropy)
            # dL/dw_k += -lambda * (-(log(w_k) + 1)) = lambda * (log(w_k) + 1)
            eps = 1e-10
            dL_dw_entropy = lambda_entropy * (np.log(w + eps) + 1)
            # Convert to dL/dz via softmax Jacobian
            weighted_sum = np.sum(dL_dw_entropy * w)
            dL_dz_entropy = w * (dL_dw_entropy - weighted_sum)
            dL_dz += dL_dz_entropy

        if lambda_dirichlet > 0:
            # Dirichlet MAP prior: log p(w) = sum_k (alpha - 1) * log(w_k)
            # We minimize -lambda * log p(w)
            # dL/dw_k = -lambda * (alpha - 1) / (w_k + eps)
            # With alpha < 1, (alpha - 1) < 0, so gradient is positive for small w
            eps = 1e-10
            dL_dw_dirichlet = -lambda_dirichlet * (alpha_dirichlet - 1.0) / (w + eps)
            # Convert to dL/dz via softmax Jacobian
            weighted_sum = np.sum(dL_dw_dirichlet * w)
            dL_dz_dirichlet = w * (dL_dw_dirichlet - weighted_sum)
            dL_dz += dL_dz_dirichlet

        # Compilation penalty (Phase III)
        compile_loss = 0.0
        if use_compile:
            from .compilation import compilation_penalty, compilation_penalty_grad_logits
            compile_loss = compilation_penalty(w, masks, device_graph, lambda_compile)
            dL_dz_compile = compilation_penalty_grad_logits(
                w, masks, device_graph, lambda_compile
            )
            dL_dz += dL_dz_compile

        # Update parameters
        model.step(dW, dv, lr=lr_params)
        mixture.step(dL_dz, lr=lr_mixture)

        # Compute metrics
        acc = float(((p > 0.5) == (y > 0.5)).mean())
        l1_dist = float(np.sum(np.abs(w - w_true)))

        # KL divergence: sum_k w_true_k * log(w_true_k / w_k)
        eps = 1e-10
        kl_div = float(np.sum(w_true * np.log((w_true + eps) / (w + eps))))

        # Record history
        history['loss'].append(float(loss))
        history['acc'].append(acc)
        history['weights'].append(w.copy())
        history['recovery_l1'].append(l1_dist)
        history['recovery_kl'].append(kl_div)
        if use_compile:
            history['loss_compile'].append(float(compile_loss))

    # Convert to arrays
    history['weights'] = np.array(history['weights'])
    history['loss'] = np.array(history['loss'])
    history['acc'] = np.array(history['acc'])
    history['recovery_l1'] = np.array(history['recovery_l1'])
    history['recovery_kl'] = np.array(history['recovery_kl'])
    if use_compile:
        history['loss_compile'] = np.array(history['loss_compile'])

    return history


def run_recovery_phase_diagram(
    n: int = 12,
    batch: int = 512,
    epochs: int = 200,
    topology_names: List[str] = None,
    noise_levels: np.ndarray = None,
    dataset_sizes: np.ndarray = None,
    seed: int = 0,
) -> Dict:
    """
    Run phase diagram sweep over noise level and dataset size.

    Returns a grid of final recovery errors.

    Parameters
    ----------
    n : int
        Number of nodes.
    batch : int
        Base batch size (used when sweeping noise only).
    epochs : int
        Training epochs per run.
    topology_names : List[str]
        Topologies to use.
    noise_levels : np.ndarray
        Noise levels to sweep.
    dataset_sizes : np.ndarray
        Dataset sizes to sweep.
    seed : int
        Base random seed.

    Returns
    -------
    results : dict
        - 'noise_levels': (N_noise,)
        - 'dataset_sizes': (N_sizes,)
        - 'recovery_l1_grid': (N_noise, N_sizes) final L1 errors
        - 'accuracy_grid': (N_noise, N_sizes) final accuracies
    """
    if topology_names is None:
        topology_names = ["chain", "ring", "star", "complete"]
    if noise_levels is None:
        noise_levels = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    if dataset_sizes is None:
        dataset_sizes = np.array([64, 128, 256, 512, 1024])

    K = len(topology_names)
    N_noise = len(noise_levels)
    N_sizes = len(dataset_sizes)

    recovery_l1_grid = np.zeros((N_noise, N_sizes))
    accuracy_grid = np.zeros((N_noise, N_sizes))

    # Sample one ground-truth mixture for all experiments
    w_true = sample_ground_truth_mixture(K, seed=seed, concentration=0.5)

    for i, noise in enumerate(noise_levels):
        for j, size in enumerate(dataset_sizes):
            # Generate data with this noise level and size
            X, y, _, _ = generate_planted_mixture_data(
                n=n,
                batch=int(size),
                w_true=w_true,
                topology_names=topology_names,
                seed=seed + i * 1000 + j,
                noise_level=noise,
            )

            # Train and recover
            history = train_mixture_recovery(
                X, y, w_true,
                topology_names=topology_names,
                n=n,
                epochs=epochs,
                seed=seed,
            )

            # Record final metrics
            recovery_l1_grid[i, j] = history['recovery_l1'][-1]
            accuracy_grid[i, j] = history['acc'][-1]

    return {
        'noise_levels': noise_levels,
        'dataset_sizes': dataset_sizes,
        'recovery_l1_grid': recovery_l1_grid,
        'accuracy_grid': accuracy_grid,
        'w_true': w_true,
    }
