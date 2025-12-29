"""
Adaptive Topology Learning (AQN) module.

Implements minimal adaptive dynamics for topology logits, treating
adaptive learning as control that responds to identified degeneracy.
Provides stability mechanisms (momentum, inertia) and constraint coupling.
"""

import numpy as np
from typing import Dict, List, Optional


def train_adaptive_topology(
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
    enable_adaptive_topology: bool = False,
    adaptive_beta: float = 0.0,
    adaptive_momentum: float = 0.0,
    adaptive_update: str = "momentum",
    enable_multi_observable: bool = False,
    lambda_aux: float = 0.0,
    aux_task: str = "graph_feature",
    aux_seed: int = 0,
) -> Dict:
    """
    Train mixture recovery with adaptive topology dynamics.

    When enable_adaptive_topology is True, applies controlled updates to
    topology logits using momentum or EMA for stability.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (batch, n).
    y : np.ndarray
        Binary labels, shape (batch,).
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
        Device graph.
    lambda_compile : float
        Compilation penalty strength.
    enable_adaptive_topology : bool
        Enable adaptive topology dynamics. Default False (no adaptive behavior).
    adaptive_beta : float
        Inertia/EMA parameter. 0 disables adaptive updates. Range [0, 1].
        Higher values = more smoothing.
    adaptive_momentum : float
        Momentum parameter for gradient updates. Range [0, 1].
    adaptive_update : str
        Update rule: "momentum" or "ema".
    enable_multi_observable : bool
        Enable multi-observable training.
    lambda_aux : float
        Auxiliary loss weight.
    aux_task : str
        Auxiliary task type.
    aux_seed : int
        Auxiliary random seed.

    Returns
    -------
    history : dict
        Training history with additional keys if adaptive is enabled:
        - 'z_logits': Logit snapshots (every 10 epochs)
        - 'adaptive_step_norm': Norm of adaptive update per epoch
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

    # Adaptive state
    if enable_adaptive_topology and adaptive_beta > 0:
        z_velocity = np.zeros(K)  # For momentum
        z_prev = mixture.z.copy()  # For EMA
    else:
        enable_adaptive_topology = False  # Force disable if beta=0

    # Multi-observable setup
    if enable_multi_observable and lambda_aux > 0:
        from .multi_observable import generate_auxiliary_labels, compute_auxiliary_loss_and_grad
        y_aux = generate_auxiliary_labels(
            X, w_true, topology_names, n, aux_task, aux_seed, include_self, normalize
        )
    else:
        enable_multi_observable = False

    # Compilation constraints
    use_compile = (
        enable_compile_constraints
        and device_graph is not None
        and lambda_compile > 0
    )

    # History
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
    if enable_multi_observable:
        history['loss_aux'] = []
        history['acc_aux'] = []
    if enable_adaptive_topology:
        history['z_logits'] = []
        history['adaptive_step_norm'] = []

    for epoch in range(epochs):
        # Get current mixture
        M = mixture.mixture_mask()
        w = mixture.weights()

        # Main task
        loss_main, dW, dv = model.loss_and_grads(X, y, M)

        # Compute dL_main/dM
        W_eff = model.W * M
        p = model.forward(X, M)
        dlogits = (p - y) / X.shape[0]
        h = X @ W_eff.T
        dH = np.outer(dlogits, model.v)
        dWeff = dH.T @ X
        dL_main_dM = compute_dL_dM(model.W, dWeff)

        # Multi-observable
        dL_dM = dL_main_dM.copy()
        loss_aux = 0.0
        if enable_multi_observable:
            loss_aux, dL_aux_dM = compute_auxiliary_loss_and_grad(
                y_aux, M, lambda_aux, aux_task
            )
            dL_dM += dL_aux_dM

        # Mixture gradient
        dL_dz = mixture.grad_z(dL_dM)

        # Regularization
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

        # Update model parameters
        model.step(dW, dv, lr=lr_params)

        # Adaptive topology update
        if enable_adaptive_topology:
            z_old = mixture.z.copy()

            if adaptive_update == "momentum":
                # Momentum update: v <- momentum * v - lr * grad, z <- z + v
                z_velocity = adaptive_momentum * z_velocity - lr_mixture * dL_dz
                mixture.z = mixture.z + z_velocity
            elif adaptive_update == "ema":
                # Standard gradient step + EMA smoothing
                z_new = mixture.z - lr_mixture * dL_dz
                # EMA: z <- (1-beta) * z_new + beta * z_prev
                mixture.z = (1.0 - adaptive_beta) * z_new + adaptive_beta * z_prev
                z_prev = mixture.z.copy()
            else:
                # Fallback to standard gradient descent
                mixture.step(dL_dz, lr=lr_mixture)

            adaptive_step_norm = float(np.linalg.norm(mixture.z - z_old))
            history['adaptive_step_norm'].append(adaptive_step_norm)

            # Snapshot logits every 10 epochs
            if epoch % 10 == 0:
                history['z_logits'].append(mixture.z.copy())
        else:
            # Standard update
            mixture.step(dL_dz, lr=lr_mixture)

        # Metrics
        acc = float(((p > 0.5) == (y > 0.5)).mean())
        l1_dist = float(np.sum(np.abs(w - w_true)))
        eps = 1e-10
        kl_div = float(np.sum(w_true * np.log((w_true + eps) / (w + eps))))

        # Record
        history['loss'].append(float(loss_main))
        history['acc'].append(acc)
        history['weights'].append(w.copy())
        history['recovery_l1'].append(l1_dist)
        history['recovery_kl'].append(kl_div)
        if use_compile:
            history['loss_compile'].append(float(compile_loss))
        if enable_multi_observable:
            history['loss_aux'].append(float(loss_aux))
            # Aux metric (MSE)
            from .multi_observable import compute_graph_features
            if y_aux.ndim == 0:
                y_aux_mean = float(y_aux)
            else:
                y_aux_mean = float(np.mean(y_aux))
            y_aux_pred = compute_graph_features(M, feature_type="triangle_proxy")[0]
            aux_mse = (y_aux_pred - y_aux_mean) ** 2
            history['acc_aux'].append(float(aux_mse))

    # Convert to arrays
    history['weights'] = np.array(history['weights'])
    history['loss'] = np.array(history['loss'])
    history['acc'] = np.array(history['acc'])
    history['recovery_l1'] = np.array(history['recovery_l1'])
    history['recovery_kl'] = np.array(history['recovery_kl'])
    if use_compile:
        history['loss_compile'] = np.array(history['loss_compile'])
    if enable_multi_observable:
        history['loss_aux'] = np.array(history['loss_aux'])
        history['acc_aux'] = np.array(history['acc_aux'])
    if enable_adaptive_topology:
        history['z_logits'] = np.array(history['z_logits'])
        history['adaptive_step_norm'] = np.array(history['adaptive_step_norm'])

    return history
