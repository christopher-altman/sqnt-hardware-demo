"""
Multi-layer SQNT network with per-layer topology mixtures.

Extends the single-layer SQNT to support L hidden layers, each with
its own learnable topology mixture.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from .graphs import make_graph_mask
from .mixture import TopologyMixture, softmax, compute_dL_dM
from .sqnt_layer import sigmoid


class SQNTMLPLayer:
    """
    Single layer of the SQNT MLP with topology-masked weights.

    Parameters
    ----------
    n_in : int
        Input dimension.
    n_out : int
        Output dimension.
    seed : int
        Random seed.
    """

    def __init__(self, n_in: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_out = n_out
        # Xavier initialization
        scale = np.sqrt(2.0 / (n_in + n_out))
        self.W = rng.standard_normal((n_out, n_in)) * scale
        self.b = np.zeros(n_out)

    def forward(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Forward pass with topology mask.

        Parameters
        ----------
        X : np.ndarray
            Input, shape (batch, n_in).
        M : np.ndarray
            Topology mask, shape (n_out, n_in).

        Returns
        -------
        h : np.ndarray
            Output, shape (batch, n_out).
        """
        W_eff = self.W * M
        return X @ W_eff.T + self.b


class SQNTMLP:
    """
    Multi-layer SQNT network with per-layer topology mixtures.

    Each hidden layer has:
    - Weight matrix W^(l)
    - Bias b^(l)
    - Topology mixture with learnable logits z^(l)

    Parameters
    ----------
    layer_sizes : List[int]
        Sizes of each layer including input and output.
        E.g., [n_input, n_hidden1, n_hidden2, n_output].
    topology_names : List[str]
        Topology names for masks.
    seed : int
        Random seed.
    include_self : bool
        Include self-loops in topology masks.
    normalize : bool
        Row-normalize topology masks.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        topology_names: List[str],
        seed: int = 0,
        include_self: bool = True,
        normalize: bool = True,
    ):
        self.layer_sizes = layer_sizes
        self.topology_names = topology_names
        self.n_layers = len(layer_sizes) - 1
        self.seed = seed

        # Build layers and mixtures
        self.layers = []
        self.mixtures = []

        rng = np.random.default_rng(seed)

        for l in range(self.n_layers):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l + 1]

            # Create layer
            layer_seed = int(rng.integers(0, 2**31))
            layer = SQNTMLPLayer(n_in, n_out, seed=layer_seed)
            self.layers.append(layer)

            # Create topology masks for this layer
            # Masks have shape (n_out, n_in) to match weight matrix
            # For non-square layers, we create masks of appropriate size
            # by taking the min dimension and tiling/cropping
            mask_size = min(n_in, n_out)
            base_masks = [make_graph_mask(name, mask_size,
                                          include_self=include_self,
                                          normalize=normalize)
                          for name in topology_names]

            # Resize masks to (n_out, n_in)
            layer_masks = []
            for m in base_masks:
                M_resized = np.zeros((n_out, n_in))
                # Tile the mask to cover the full matrix
                for i in range(n_out):
                    for j in range(n_in):
                        M_resized[i, j] = m[i % mask_size, j % mask_size]
                layer_masks.append(M_resized)

            # Create mixture for this layer
            mix_seed = int(rng.integers(0, 2**31))
            mixture = TopologyMixture(layer_masks, seed=mix_seed)
            self.mixtures.append(mixture)

        # Output layer (final classification)
        self.v = rng.standard_normal(layer_sizes[-1]) * 0.1

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through all layers.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (batch, n_input).

        Returns
        -------
        p : np.ndarray
            Output probabilities, shape (batch,).
        hiddens : List[np.ndarray]
            Hidden activations for each layer.
        """
        hiddens = [X]
        h = X

        for l in range(self.n_layers):
            M = self.mixtures[l].mixture_mask()
            h = self.layers[l].forward(h, M)
            h = np.tanh(h)  # Activation
            hiddens.append(h)

        # Final output
        logits = h @ self.v
        p = sigmoid(logits)

        return p, hiddens

    def loss_and_grads(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Compute loss and gradients for all parameters.

        Returns
        -------
        loss : float
            Binary cross-entropy loss.
        dWs : List[np.ndarray]
            Gradients for each layer's weights.
        dbs : List[np.ndarray]
            Gradients for each layer's biases.
        dL_dzs : List[np.ndarray]
            Gradients for each layer's mixture logits.
        dv : np.ndarray
            Gradient for output vector.
        """
        batch = X.shape[0]

        # Forward pass
        p, hiddens = self.forward(X)

        # Loss
        eps = 1e-9
        loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()

        # Backward pass
        dlogits = (p - y) / batch  # (batch,)

        # Gradient for output vector
        dv = hiddens[-1].T @ dlogits  # (n_last,)

        # Backprop through layers
        dh = np.outer(dlogits, self.v)  # (batch, n_last)

        dWs = []
        dbs = []
        dL_dzs = []

        for l in range(self.n_layers - 1, -1, -1):
            # Get this layer's info
            layer = self.layers[l]
            mixture = self.mixtures[l]
            M = mixture.mixture_mask()
            h_prev = hiddens[l]  # Input to this layer
            h_curr = hiddens[l + 1]  # Output of this layer (after activation)

            # Gradient through tanh: dL/d(pre_act) = dL/dh * (1 - tanh^2)
            d_pre_act = dh * (1 - h_curr ** 2)  # (batch, n_out)

            # Gradient for bias
            db = d_pre_act.sum(axis=0)  # (n_out,)
            dbs.insert(0, db)

            # Gradient for effective weights
            dWeff = d_pre_act.T @ h_prev  # (n_out, n_in)

            # Gradient for weights (masked)
            dW = dWeff * M
            dWs.insert(0, dW)

            # Gradient for mixture logits
            dL_dM = dWeff * layer.W
            dL_dz = mixture.grad_z(dL_dM)
            dL_dzs.insert(0, dL_dz)

            # Backprop to previous layer
            W_eff = layer.W * M
            dh = d_pre_act @ W_eff  # (batch, n_in)

        return loss, dWs, dbs, dL_dzs, dv

    def step(
        self,
        dWs: List[np.ndarray],
        dbs: List[np.ndarray],
        dL_dzs: List[np.ndarray],
        dv: np.ndarray,
        lr_params: float = 0.1,
        lr_mixture: float = 0.1,
    ):
        """Update all parameters via gradient descent."""
        for l in range(self.n_layers):
            self.layers[l].W -= lr_params * dWs[l]
            self.layers[l].b -= lr_params * dbs[l]
            self.mixtures[l].step(dL_dzs[l], lr=lr_mixture)

        self.v -= lr_params * dv

    def get_all_weights(self) -> List[np.ndarray]:
        """Return mixture weights for all layers."""
        return [m.weights() for m in self.mixtures]


def train_mlp_mixture(
    X: np.ndarray,
    y: np.ndarray,
    layer_sizes: List[int],
    topology_names: List[str],
    epochs: int = 200,
    lr_params: float = 0.1,
    lr_mixture: float = 0.1,
    seed: int = 0,
    w_true_per_layer: Optional[List[np.ndarray]] = None,
) -> Dict:
    """
    Train a multi-layer SQNT network.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (batch, n_input).
    y : np.ndarray
        Binary labels, shape (batch,).
    layer_sizes : List[int]
        Layer sizes including input and output.
    topology_names : List[str]
        Topology names.
    epochs : int
        Training epochs.
    lr_params : float
        Learning rate for weights.
    lr_mixture : float
        Learning rate for mixture logits.
    seed : int
        Random seed.
    w_true_per_layer : Optional[List[np.ndarray]]
        If provided, ground-truth weights for tracking recovery.

    Returns
    -------
    history : dict
        Training history.
    """
    model = SQNTMLP(
        layer_sizes=layer_sizes,
        topology_names=topology_names,
        seed=seed,
    )

    n_layers = model.n_layers
    K = len(topology_names)

    history = {
        'loss': [],
        'acc': [],
        'weights_per_layer': [[] for _ in range(n_layers)],
    }

    if w_true_per_layer is not None:
        history['recovery_l1_per_layer'] = [[] for _ in range(n_layers)]

    for epoch in range(epochs):
        loss, dWs, dbs, dL_dzs, dv = model.loss_and_grads(X, y)
        model.step(dWs, dbs, dL_dzs, dv, lr_params=lr_params, lr_mixture=lr_mixture)

        # Compute accuracy
        p, _ = model.forward(X)
        acc = float(((p > 0.5) == (y > 0.5)).mean())

        history['loss'].append(float(loss))
        history['acc'].append(acc)

        # Record weights for each layer
        for l in range(n_layers):
            w = model.mixtures[l].weights()
            history['weights_per_layer'][l].append(w.copy())

            if w_true_per_layer is not None:
                l1 = float(np.sum(np.abs(w - w_true_per_layer[l])))
                history['recovery_l1_per_layer'][l].append(l1)

    # Convert to arrays
    history['loss'] = np.array(history['loss'])
    history['acc'] = np.array(history['acc'])
    for l in range(n_layers):
        history['weights_per_layer'][l] = np.array(history['weights_per_layer'][l])
        if w_true_per_layer is not None:
            history['recovery_l1_per_layer'][l] = np.array(history['recovery_l1_per_layer'][l])

    return history, model
