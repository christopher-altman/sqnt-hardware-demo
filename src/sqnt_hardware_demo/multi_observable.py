"""
Multi-observable identifiability protocols for SQNT.

Implements auxiliary observables and joint loss training to break topology
confusability. Phase IV adds at least one additional observable channel
that is topology-informative in a way that improves support recovery.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def compute_graph_features(mask: np.ndarray, feature_type: str = "triangle_proxy") -> np.ndarray:
    """
    Compute topology-informative graph features from a mask.

    These features are designed to differentiate topologies that may be
    confusable under the main observable. Returns deterministic features
    that depend on the mask structure.

    Parameters
    ----------
    mask : np.ndarray
        Topology mask of shape (n, n).
    feature_type : str
        Type of graph feature to compute:
        - "triangle_proxy": Count of approximate triangles (3-cycles)
        - "hubness_proxy": Variance of degree centrality
        - "spectral_proxy": Spectral radius approximation
        - "path_proxy": Average shortest path proxy

    Returns
    -------
    features : np.ndarray
        Feature vector, shape depends on feature_type. For most types,
        returns a scalar expanded to shape (1,).
    """
    n = mask.shape[0]
    eps = 1e-10

    if feature_type == "triangle_proxy":
        # Count triangles: A^3 diagonal counts closed 3-paths
        # For weighted/normalized masks, this is a proxy
        A = (mask + mask.T) / 2.0  # Symmetrize
        A_cubed = A @ A @ A
        triangle_count = np.trace(A_cubed) / 6.0  # Each triangle counted 6 times
        return np.array([float(triangle_count)])

    elif feature_type == "hubness_proxy":
        # Variance of row sums (degree centrality proxy)
        row_sums = mask.sum(axis=1)
        hubness = float(np.var(row_sums))
        return np.array([hubness])

    elif feature_type == "spectral_proxy":
        # Spectral radius approximation via power iteration (cheap)
        A = (mask + mask.T) / 2.0
        v = np.ones(n) / np.sqrt(n)
        for _ in range(5):  # Few iterations for speed
            v = A @ v
            norm_v = np.linalg.norm(v)
            if norm_v > eps:
                v = v / norm_v
        spectral_radius = float(np.linalg.norm(A @ v))
        return np.array([spectral_radius])

    elif feature_type == "path_proxy":
        # Average shortest path proxy: sum of (A^k)_ij for small k
        # This approximates connectivity at different scales
        A = (mask + mask.T) / 2.0
        A_power = A.copy()
        total_reachability = A.copy()
        for k in range(2, 4):  # k=2,3
            A_power = A_power @ A
            total_reachability += A_power / (k + 1.0)
        path_metric = float(total_reachability.sum() / (n * n + eps))
        return np.array([path_metric])

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def generate_auxiliary_labels(
    X: np.ndarray,
    w_true: np.ndarray,
    topology_names: List[str],
    n: int,
    aux_task: str = "graph_feature",
    aux_seed: int = 0,
    include_self: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate auxiliary labels for multi-observable training.

    The auxiliary channel is topology-informative: it depends on the
    true mixture in a way that breaks confusability.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (batch, n). Not used for graph_feature task.
    w_true : np.ndarray
        Ground-truth mixture weights, shape (K,).
    topology_names : List[str]
        List of topology names.
    n : int
        Number of nodes.
    aux_task : str
        Auxiliary task type:
        - "graph_feature": Predict a graph feature from the true mixture.
    aux_seed : int
        Random seed for noise (if any).
    include_self : bool
        Include self-loops in masks.
    normalize : bool
        Row-normalize masks.

    Returns
    -------
    y_aux : np.ndarray
        Auxiliary labels, shape (batch,) or (batch, d_aux).
    """
    from .graphs import make_graph_mask

    batch = X.shape[0]

    if aux_task == "graph_feature":
        # Compute true mixture mask
        masks = [make_graph_mask(name, n, include_self=include_self, normalize=normalize)
                 for name in topology_names]
        K = len(masks)
        M_true = np.zeros((n, n))
        for k in range(K):
            M_true += w_true[k] * masks[k]

        # Compute graph features (deterministic, depends on true topology)
        features = compute_graph_features(M_true, feature_type="triangle_proxy")

        # Replicate for all samples (same label for all in batch)
        # Add small noise for variation across samples
        rng = np.random.default_rng(aux_seed)
        y_aux = np.tile(features, (batch, 1))
        noise = rng.normal(0, 0.01, size=y_aux.shape)
        y_aux = y_aux + noise

        # Return scalar labels if d_aux = 1
        if y_aux.shape[1] == 1:
            y_aux = y_aux.ravel()

        return y_aux

    else:
        raise ValueError(f"Unknown aux_task: {aux_task}")


def compute_auxiliary_loss_and_grad(
    y_aux_true: np.ndarray,
    M: np.ndarray,
    lambda_aux: float,
    aux_task: str = "graph_feature",
) -> Tuple[float, np.ndarray]:
    """
    Compute auxiliary loss and its gradient w.r.t. the mixture mask M.

    For regression tasks, uses MSE loss.

    Parameters
    ----------
    y_aux_true : np.ndarray
        True auxiliary labels, shape (batch,) or scalar.
    M : np.ndarray
        Current mixture mask, shape (n, n).
    lambda_aux : float
        Auxiliary loss weight.
    aux_task : str
        Auxiliary task type.

    Returns
    -------
    loss_aux : float
        Auxiliary loss (unweighted).
    dL_aux_dM : np.ndarray
        Gradient of weighted auxiliary loss w.r.t. M, shape (n, n).
        Already scaled by lambda_aux.
    """
    if lambda_aux == 0:
        return 0.0, np.zeros_like(M)

    # Compute predicted features from current mask
    if aux_task == "graph_feature":
        y_aux_pred_vec = compute_graph_features(M, feature_type="triangle_proxy")
        y_aux_pred = y_aux_pred_vec[0]  # Scalar

        # True label (average over batch if multiple samples)
        if y_aux_true.ndim == 0:
            y_aux_mean = float(y_aux_true)
        else:
            y_aux_mean = float(np.mean(y_aux_true))

        # MSE loss
        error = y_aux_pred - y_aux_mean
        loss_aux = 0.5 * error ** 2

        # Gradient: dL/dy_pred = error, then chain through feature computation
        # For triangle_proxy: y = trace(A^3) / 6 where A = (M + M.T)/2
        # dy/dA_ij = d(trace(A^3))/dA_ij / 6
        # trace(A^3) = sum_ijk A_ij A_jk A_ki
        # d(trace)/dA_ij = (A^2)_ji + (A^2)_ij + A_ij (from symmetry)
        # For simplicity, use numerical gradient or approximate

        # Approximate gradient via finite differences (cheap for small n)
        n = M.shape[0]
        A = (M + M.T) / 2.0
        A_sq = A @ A
        # d(trace(A^3))/dA_ij ≈ 3 * (A^2)_ji (using symmetry)
        dy_dA = 3.0 * A_sq.T / 6.0  # shape (n, n)

        # Chain rule: dL/dA = (dL/dy) * (dy/dA)
        dL_dA = error * dy_dA

        # Convert dL/dA to dL/dM (since A = (M + M.T)/2)
        dL_dM = (dL_dA + dL_dA.T) / 2.0

        # Scale by lambda_aux
        dL_dM = lambda_aux * dL_dM

        return float(loss_aux), dL_dM

    else:
        raise ValueError(f"Unknown aux_task: {aux_task}")


def train_multi_observable(
    X: np.ndarray,
    y: np.ndarray,
    y_aux: np.ndarray,
    w_true: np.ndarray,
    topology_names: List[str],
    n: int,
    epochs: int = 300,
    lr_params: float = 0.2,
    lr_mixture: float = 0.15,
    seed: int = 0,
    include_self: bool = True,
    normalize: bool = True,
    lambda_aux: float = 0.1,
    aux_task: str = "graph_feature",
    lambda_sparsity: float = 0.0,
    lambda_entropy: float = 0.0,
    lambda_dirichlet: float = 0.0,
    alpha_dirichlet: float = 0.3,
    enable_compile_constraints: bool = False,
    device_graph: Optional[dict] = None,
    lambda_compile: float = 0.0,
) -> Dict:
    """
    Train mixture recovery with multi-observable protocol.

    Jointly optimizes main task + auxiliary task to improve identifiability.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (batch, n).
    y : np.ndarray
        Main task labels (binary), shape (batch,).
    y_aux : np.ndarray
        Auxiliary task labels, shape (batch,) or (batch, d_aux).
    w_true : np.ndarray
        Ground-truth mixture weights, shape (K,).
    topology_names : List[str]
        List of topology names.
    n : int
        Number of nodes.
    epochs : int
        Training epochs.
    lr_params : float
        Learning rate for W, v.
    lr_mixture : float
        Learning rate for mixture logits.
    seed : int
        Random seed.
    include_self : bool
        Include self-loops.
    normalize : bool
        Row-normalize masks.
    lambda_aux : float
        Auxiliary loss weight.
    aux_task : str
        Auxiliary task type.
    lambda_sparsity : float
        Sparsity penalty.
    lambda_entropy : float
        Entropy penalty.
    lambda_dirichlet : float
        Dirichlet prior strength.
    alpha_dirichlet : float
        Dirichlet concentration.
    enable_compile_constraints : bool
        Enable compilation penalty.
    device_graph : dict, optional
        Device graph for compilation.
    lambda_compile : float
        Compilation penalty strength.

    Returns
    -------
    history : dict
        Training history with keys:
        - 'loss': Main task loss
        - 'acc': Main task accuracy
        - 'loss_aux': Auxiliary loss
        - 'acc_aux': Auxiliary metric (e.g., MSE for regression)
        - 'weights': Mixture weights over time
        - 'weights_true': Ground truth weights
        - 'recovery_l1': L1 distance to true weights
        - 'recovery_kl': KL divergence
    """
    from .graphs import make_graph_mask
    from .sqnt_layer import SQNTLayer
    from .mixture import TopologyMixture, compute_dL_dM

    # Build topology masks
    masks = [make_graph_mask(name, n, include_self=include_self, normalize=normalize)
             for name in topology_names]
    K = len(masks)

    # Initialize model and mixture
    model = SQNTLayer(n=n, seed=seed)
    mixture = TopologyMixture(masks, seed=seed)

    # Check compilation constraints
    use_compile = (
        enable_compile_constraints
        and device_graph is not None
        and lambda_compile > 0
    )

    # History
    history = {
        'loss': [],
        'acc': [],
        'loss_aux': [],
        'acc_aux': [],
        'weights': [],
        'weights_true': w_true.copy(),
        'recovery_l1': [],
        'recovery_kl': [],
    }
    if use_compile:
        history['loss_compile'] = []

    for epoch in range(epochs):
        # Get current mixture
        M = mixture.mixture_mask()
        w = mixture.weights()

        # Main task forward pass
        loss_main, dW, dv = model.loss_and_grads(X, y, M)

        # Compute dL_main/dM
        W_eff = model.W * M
        p = model.forward(X, M)
        dlogits = (p - y) / X.shape[0]
        h = X @ W_eff.T
        dH = np.outer(dlogits, model.v)
        dWeff = dH.T @ X
        dL_main_dM = compute_dL_dM(model.W, dWeff)

        # Auxiliary task
        loss_aux, dL_aux_dM = compute_auxiliary_loss_and_grad(
            y_aux, M, lambda_aux, aux_task
        )

        # Total dL/dM
        dL_dM = dL_main_dM + dL_aux_dM

        # Mixture gradient
        dL_dz = mixture.grad_z(dL_dM)

        # Regularization (same as train_mixture_recovery)
        if lambda_sparsity > 0:
            dL_dM_sparse = lambda_sparsity * np.ones_like(M)
            dL_dz += mixture.grad_z(dL_dM_sparse)

        if lambda_entropy != 0:
            eps = 1e-10
            dL_dw_entropy = lambda_entropy * (np.log(w + eps) + 1)
            weighted_sum = np.sum(dL_dw_entropy * w)
            dL_dz_entropy = w * (dL_dw_entropy - weighted_sum)
            dL_dz += dL_dz_entropy

        if lambda_dirichlet > 0:
            eps = 1e-10
            dL_dw_dirichlet = -lambda_dirichlet * (alpha_dirichlet - 1.0) / (w + eps)
            weighted_sum = np.sum(dL_dw_dirichlet * w)
            dL_dz_dirichlet = w * (dL_dw_dirichlet - weighted_sum)
            dL_dz += dL_dz_dirichlet

        compile_loss = 0.0
        if use_compile:
            from .compilation import compilation_penalty, compilation_penalty_grad_logits
            compile_loss = compilation_penalty(w, masks, device_graph, lambda_compile)
            dL_dz_compile = compilation_penalty_grad_logits(
                w, masks, device_graph, lambda_compile
            )
            dL_dz += dL_dz_compile

        # Update
        model.step(dW, dv, lr=lr_params)
        mixture.step(dL_dz, lr=lr_mixture)

        # Metrics
        acc = float(((p > 0.5) == (y > 0.5)).mean())

        # Auxiliary metric (MSE for regression)
        if y_aux.ndim == 0:
            y_aux_mean = float(y_aux)
        else:
            y_aux_mean = float(np.mean(y_aux))
        y_aux_pred = compute_graph_features(M, feature_type="triangle_proxy")[0]
        aux_mse = (y_aux_pred - y_aux_mean) ** 2

        l1_dist = float(np.sum(np.abs(w - w_true)))
        eps = 1e-10
        kl_div = float(np.sum(w_true * np.log((w_true + eps) / (w + eps))))

        # Record
        history['loss'].append(float(loss_main))
        history['acc'].append(acc)
        history['loss_aux'].append(float(loss_aux))
        history['acc_aux'].append(float(aux_mse))
        history['weights'].append(w.copy())
        history['recovery_l1'].append(l1_dist)
        history['recovery_kl'].append(kl_div)
        if use_compile:
            history['loss_compile'].append(float(compile_loss))

    # Convert to arrays
    history['weights'] = np.array(history['weights'])
    history['loss'] = np.array(history['loss'])
    history['acc'] = np.array(history['acc'])
    history['loss_aux'] = np.array(history['loss_aux'])
    history['acc_aux'] = np.array(history['acc_aux'])
    history['recovery_l1'] = np.array(history['recovery_l1'])
    history['recovery_kl'] = np.array(history['recovery_kl'])
    if use_compile:
        history['loss_compile'] = np.array(history['loss_compile'])

    return history
