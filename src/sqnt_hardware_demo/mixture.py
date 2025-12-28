"""
Topology mixture module for learned superposition of graph topologies.

Implements a differentiable mixture of K graph topology masks with
analytically computed gradients for gradient descent training.
"""

import numpy as np
from typing import List, Tuple, Dict


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)


class TopologyMixture:
    """
    Learnable mixture over K graph topology masks.

    Maintains trainable logits z where mixture weights w = softmax(z).
    The effective mask is M = sum_k w_k * M_k.

    Parameters
    ----------
    masks : List[np.ndarray]
        List of K topology masks, each of shape (n_out, n_in) or (n, n).
    seed : int
        Random seed for logit initialization.
    """

    def __init__(self, masks: List[np.ndarray], seed: int = 0):
        self.masks = [m.copy() for m in masks]
        self.K = len(masks)
        self.shape = masks[0].shape  # Can be non-square for MLP layers

        # Validate mask shapes (all must match first)
        for i, m in enumerate(masks):
            if m.shape != self.shape:
                raise ValueError(f"Mask {i} has shape {m.shape}, expected {self.shape}")

        # Initialize logits (uniform initialization -> equal weights)
        rng = np.random.default_rng(seed)
        self.z = rng.standard_normal(self.K) * 0.01

    def weights(self) -> np.ndarray:
        """Return mixture weights w = softmax(z)."""
        return softmax(self.z)

    def mixture_mask(self) -> np.ndarray:
        """Return effective mask M = sum_k w_k * M_k."""
        w = self.weights()
        M = np.zeros(self.shape)
        for k in range(self.K):
            M += w[k] * self.masks[k]
        return M

    def grad_z(self, dL_dM: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. logits z.

        Given dL/dM (the gradient of loss w.r.t. the mixture mask),
        compute dL/dz via chain rule:

        1. dL/dw_k = sum_{ij} (dL/dM_ij) * (M_k)_{ij}
        2. dw_k/dz_l = w_k * (delta_{kl} - w_l)  (softmax Jacobian)
        3. dL/dz_l = sum_k (dL/dw_k) * dw_k/dz_l

        Parameters
        ----------
        dL_dM : np.ndarray
            Gradient of loss w.r.t. the mixture mask, shape (n, n).

        Returns
        -------
        dL_dz : np.ndarray
            Gradient of loss w.r.t. logits, shape (K,).
        """
        w = self.weights()

        # Step 1: dL/dw_k = sum_{ij} (dL/dM_ij) * (M_k)_{ij}
        dL_dw = np.array([np.sum(dL_dM * self.masks[k]) for k in range(self.K)])

        # Step 2-3: Apply softmax Jacobian
        # dL/dz_l = sum_k dL/dw_k * w_k * (delta_{kl} - w_l)
        #         = w_l * (dL/dw_l - sum_k dL/dw_k * w_k)
        weighted_sum = np.sum(dL_dw * w)
        dL_dz = w * (dL_dw - weighted_sum)

        return dL_dz

    def step(self, dL_dz: np.ndarray, lr: float = 0.1):
        """Update logits via gradient descent."""
        self.z -= lr * dL_dz


def compute_dL_dM(W: np.ndarray, dL_dWeff: np.ndarray) -> np.ndarray:
    """
    Compute gradient of loss w.r.t. mask M.

    Since W_eff = W * M (element-wise), we have:
    dL/dM_ij = dL/dW_eff_ij * W_ij

    Parameters
    ----------
    W : np.ndarray
        Weight matrix, shape (n, n).
    dL_dWeff : np.ndarray
        Gradient of loss w.r.t. effective weight matrix, shape (n, n).

    Returns
    -------
    dL_dM : np.ndarray
        Gradient of loss w.r.t. mask, shape (n, n).
    """
    return dL_dWeff * W


def train_learned_mixture(
    X: np.ndarray,
    y: np.ndarray,
    topology_names: List[str],
    n: int,
    epochs: int = 200,
    lr_params: float = 0.2,
    lr_mixture: float = 0.1,
    seed: int = 0,
    include_self: bool = True,
    normalize: bool = True
) -> Tuple[float, Dict]:
    """
    Train a model with learned topology mixture.

    Jointly optimizes:
    - Weight matrix W and output vector v (model parameters)
    - Mixture logits z (topology superposition)

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (batch, n).
    y : np.ndarray
        Binary labels, shape (batch,).
    topology_names : List[str]
        List of topology names (e.g., ["chain", "ring", "star", "complete"]).
    n : int
        Number of nodes.
    epochs : int
        Number of training epochs.
    lr_params : float
        Learning rate for W and v.
    lr_mixture : float
        Learning rate for mixture logits z.
    seed : int
        Random seed for reproducibility.
    include_self : bool
        Whether to include self-loops in topology masks.
    normalize : bool
        Whether to row-normalize topology masks.

    Returns
    -------
    final_acc : float
        Final training accuracy.
    history : dict
        Training history with keys:
        - 'loss': list of loss values per epoch
        - 'acc': list of accuracy values per epoch
        - 'weights': list of mixture weight arrays per epoch
    """
    from .graphs import make_graph_mask
    from .sqnt_layer import SQNTLayer

    # Build topology masks
    masks = [make_graph_mask(name, n, include_self=include_self, normalize=normalize)
             for name in topology_names]

    # Initialize model and mixture
    model = SQNTLayer(n=n, seed=seed)
    mixture = TopologyMixture(masks, seed=seed)

    # Training history
    history = {'loss': [], 'acc': [], 'weights': []}

    for epoch in range(epochs):
        # Get current mixture mask
        M = mixture.mixture_mask()

        # Forward pass and compute loss/gradients for W, v
        loss, dW, dv = model.loss_and_grads(X, y, M)

        # Compute gradient w.r.t. mixture mask
        # We need dL/dW_eff to compute dL/dM
        # From sqnt_layer: dW = dWeff * M, so dWeff = dW / M (where M != 0)
        # Actually, we need to recompute dWeff from the forward pass
        # Let's compute it directly
        W_eff = model.W * M
        p = model.forward(X, M)
        dlogits = (p - y) / X.shape[0]
        h = X @ W_eff.T
        dH = np.outer(dlogits, model.v)
        dWeff = dH.T @ X

        # Now compute dL/dM
        dL_dM = compute_dL_dM(model.W, dWeff)

        # Compute gradient for mixture logits
        dL_dz = mixture.grad_z(dL_dM)

        # Update parameters
        model.step(dW, dv, lr=lr_params)
        mixture.step(dL_dz, lr=lr_mixture)

        # Record history
        acc = float(((p > 0.5) == (y > 0.5)).mean())
        history['loss'].append(float(loss))
        history['acc'].append(acc)
        history['weights'].append(mixture.weights().copy())

    # Final accuracy
    M_final = mixture.mixture_mask()
    p_final = model.forward(X, M_final)
    final_acc = float(((p_final > 0.5) == (y > 0.5)).mean())

    return final_acc, history


def train_fixed_topology(
    X: np.ndarray,
    y: np.ndarray,
    topology_name: str,
    n: int,
    epochs: int = 200,
    lr: float = 0.2,
    seed: int = 0,
    include_self: bool = True,
    normalize: bool = True
) -> Tuple[float, Dict]:
    """
    Train a model with a single fixed topology.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (batch, n).
    y : np.ndarray
        Binary labels, shape (batch,).
    topology_name : str
        Topology name (e.g., "chain", "ring", "star", "complete").
    n : int
        Number of nodes.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    seed : int
        Random seed for reproducibility.
    include_self : bool
        Whether to include self-loops.
    normalize : bool
        Whether to row-normalize mask.

    Returns
    -------
    final_acc : float
        Final training accuracy.
    history : dict
        Training history with keys 'loss', 'acc'.
    """
    from .graphs import make_graph_mask
    from .sqnt_layer import SQNTLayer

    M = make_graph_mask(topology_name, n, include_self=include_self, normalize=normalize)
    model = SQNTLayer(n=n, seed=seed)

    history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        loss, dW, dv = model.loss_and_grads(X, y, M)
        model.step(dW, dv, lr=lr)

        p = model.forward(X, M)
        acc = float(((p > 0.5) == (y > 0.5)).mean())
        history['loss'].append(float(loss))
        history['acc'].append(acc)

    p_final = model.forward(X, M)
    final_acc = float(((p_final > 0.5) == (y > 0.5)).mean())

    return final_acc, history


def train_random_mixture(
    X: np.ndarray,
    y: np.ndarray,
    topology_names: List[str],
    n: int,
    epochs: int = 200,
    lr: float = 0.2,
    seed: int = 0,
    include_self: bool = True,
    normalize: bool = True
) -> Tuple[float, Dict, np.ndarray]:
    """
    Train a model with a random (frozen) topology mixture.

    The mixture weights are sampled once at initialization and kept fixed.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (batch, n).
    y : np.ndarray
        Binary labels, shape (batch,).
    topology_names : List[str]
        List of topology names.
    n : int
        Number of nodes.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    seed : int
        Random seed for reproducibility.
    include_self : bool
        Whether to include self-loops.
    normalize : bool
        Whether to row-normalize masks.

    Returns
    -------
    final_acc : float
        Final training accuracy.
    history : dict
        Training history with keys 'loss', 'acc'.
    frozen_weights : np.ndarray
        The frozen mixture weights used.
    """
    from .graphs import make_graph_mask
    from .sqnt_layer import SQNTLayer

    # Build topology masks
    masks = [make_graph_mask(name, n, include_self=include_self, normalize=normalize)
             for name in topology_names]
    K = len(masks)

    # Sample random mixture weights (deterministic via seed)
    rng = np.random.default_rng(seed + 1000)  # offset to differ from model seed
    raw_weights = rng.random(K)
    frozen_weights = raw_weights / raw_weights.sum()

    # Compute fixed mixture mask
    M = np.zeros((n, n))
    for k in range(K):
        M += frozen_weights[k] * masks[k]

    model = SQNTLayer(n=n, seed=seed)

    history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        loss, dW, dv = model.loss_and_grads(X, y, M)
        model.step(dW, dv, lr=lr)

        p = model.forward(X, M)
        acc = float(((p > 0.5) == (y > 0.5)).mean())
        history['loss'].append(float(loss))
        history['acc'].append(acc)

    p_final = model.forward(X, M)
    final_acc = float(((p_final > 0.5) == (y > 0.5)).mean())

    return final_acc, history, frozen_weights
