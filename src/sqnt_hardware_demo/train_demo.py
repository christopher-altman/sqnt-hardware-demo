import numpy as np
from .graphs import make_graph_mask
from .sqnt_layer import SQNTLayer

def make_synthetic(n: int, batch: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((batch, n))

    M_true = make_graph_mask("ring", n, include_self=True, normalize=True)
    W_true = (rng.standard_normal((n, n)) * 0.2) * M_true
    v_true = rng.standard_normal((n,)) * 0.2

    logits = (X @ W_true.T) @ v_true
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    y = (p > 0.5).astype(float)

    flip = rng.random(batch) < 0.05
    y[flip] = 1.0 - y[flip]
    return X, y

def train_for_alpha(alpha: float, n: int = 12, epochs: int = 200, lr: float = 0.2, seed: int = 0,
                    topo0: str = "chain", topo1: str = "complete"):
    X, y = make_synthetic(n=n, batch=512, seed=seed)
    model = SQNTLayer(n=n, seed=seed)

    M0 = make_graph_mask(topo0, n, include_self=True, normalize=True)
    M1 = make_graph_mask(topo1, n, include_self=True, normalize=True)
    M = (1.0 - alpha) * M0 + alpha * M1

    for _ in range(epochs):
        loss, dW, dv = model.loss_and_grads(X, y, M)
        model.step(dW, dv, lr=lr)

    p = model.forward(X, M)
    return float(((p > 0.5) == (y > 0.5)).mean())

def sweep_alphas(alphas: np.ndarray, n: int = 12, seed: int = 0, topo0: str = "chain", topo1: str = "complete"):
    return np.array([train_for_alpha(float(a), n=n, seed=seed, topo0=topo0, topo1=topo1) for a in alphas], dtype=float)
