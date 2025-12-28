import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

class SQNTLayer:
    """Tiny operator-space layer with topology spatialization: W_eff = W ⊙ M."""

    def __init__(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.W = 0.1 * rng.standard_normal((n, n))
        self.v = 0.1 * rng.standard_normal((n,))

    def forward(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        W_eff = self.W * M
        h = X @ W_eff.T
        logits = h @ self.v
        return sigmoid(logits)

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray, M: np.ndarray):
        p = self.forward(X, M)
        eps = 1e-9
        loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()

        dlogits = (p - y) / X.shape[0]
        W_eff = self.W * M
        h = X @ W_eff.T

        dv = h.T @ dlogits
        dH = np.outer(dlogits, self.v)
        dWeff = dH.T @ X
        dW = dWeff * M
        return loss, dW, dv

    def step(self, dW: np.ndarray, dv: np.ndarray, lr: float = 0.1):
        self.W -= lr * dW
        self.v -= lr * dv
