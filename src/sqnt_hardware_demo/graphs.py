import numpy as np

def _adj_chain(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A

def _adj_ring(n: int) -> np.ndarray:
    A = _adj_chain(n)
    A[0, n - 1] = 1.0
    A[n - 1, 0] = 1.0
    return A

def _adj_star(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    hub = 0
    for i in range(1, n):
        A[hub, i] = 1.0
        A[i, hub] = 1.0
    return A

def _adj_complete(n: int) -> np.ndarray:
    A = np.ones((n, n), dtype=float)
    np.fill_diagonal(A, 0.0)
    return A

def make_graph_mask(kind: str, n: int, include_self: bool = False, normalize: bool = True) -> np.ndarray:
    """Return a topology mask M for a small graph."""
    kind = kind.lower().strip()
    if kind == "chain":
        M = _adj_chain(n)
    elif kind == "ring":
        M = _adj_ring(n)
    elif kind == "star":
        M = _adj_star(n)
    elif kind == "complete":
        M = _adj_complete(n)
    else:
        raise ValueError(f"Unknown kind={kind!r}. Expected chain/ring/star/complete.")

    if include_self:
        M = M + np.eye(n)

    if normalize:
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        M = M / row_sums

    return M
