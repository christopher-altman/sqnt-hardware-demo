"""
Compilation-aware constraints for SQNT topology mixture recovery.

Provides device graph loading, adjacency conversion, and routing/SWAP
overhead penalties to incorporate hardware connectivity constraints
into the mixture recovery objective.
"""

import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional


def resolve_input_path(path: str, input_type: str = "device_graph") -> str:
    """
    Resolve input file path with fallback search order.

    Search order:
    1. Path as-given (absolute or relative to cwd)
    2. Canonical folder (device_graphs/ or noise_models/)
    3. Legacy phase3_inputs/ folder (backward compatibility)

    Parameters
    ----------
    path : str
        Path to input file (can be full path or just basename)
    input_type : str
        Type of input: "device_graph" or "noise_model"

    Returns
    -------
    resolved_path : str
        Absolute path to the resolved file

    Raises
    ------
    FileNotFoundError
        If file cannot be found in any search location
    """
    # Get repo root (assume this file is in src/sqnt_hardware_demo/)
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent.parent.parent

    # Canonical folder mapping
    canonical_folders = {
        "device_graph": "device_graphs",
        "noise_model": "noise_models",
    }
    canonical_folder = canonical_folders.get(input_type, "device_graphs")

    # Search locations in order
    search_paths = [
        Path(path),  # As-given
        repo_root / canonical_folder / Path(path).name,  # Canonical folder + basename
        repo_root / "phase3_inputs" / Path(path).name,  # Legacy compat
    ]

    for search_path in search_paths:
        if search_path.exists() and search_path.is_file():
            return str(search_path.resolve())

    # Not found - raise with helpful message
    raise FileNotFoundError(
        f"Could not find {input_type} file: {path}\n"
        f"Searched:\n" + "\n".join(f"  - {p}" for p in search_paths)
    )


def load_device_graph(path: str) -> dict:
    """
    Load a device graph from a JSON file.

    Supports flexible path resolution:
    - Full path (absolute or relative)
    - Basename only (searches device_graphs/ then phase3_inputs/)

    Expected schema:
    {
        "name": "device_name",
        "nodes": [0, 1, 2, ...],
        "edges": [[0, 1], [1, 2], ...],
        "metadata": { ... optional ... }
    }

    Parameters
    ----------
    path : str
        Path to JSON file (full path or basename).

    Returns
    -------
    device_graph : dict
        Parsed device graph dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required keys ('nodes', 'edges') are missing.
    """
    resolved_path = resolve_input_path(path, input_type="device_graph")

    with open(resolved_path, 'r') as f:
        data = json.load(f)

    # Handle two schemas:
    # Schema 1 (original): {"nodes": [...], "edges": [...]}
    # Schema 2 (phase3): {"n_qubits": N, "edges": [...]}

    if 'edges' not in data:
        raise KeyError("Device graph JSON must contain 'edges' key")

    # If nodes not provided, infer from n_qubits or edges
    if 'nodes' not in data:
        if 'n_qubits' in data:
            data['nodes'] = list(range(data['n_qubits']))
        else:
            # Infer from edges
            all_nodes = set()
            for edge in data['edges']:
                all_nodes.update(edge)
            data['nodes'] = sorted(all_nodes)

    return data


def device_graph_to_adjacency(device_graph: dict, n: int) -> np.ndarray:
    """
    Convert a device graph to an adjacency matrix of size n x n.

    Parameters
    ----------
    device_graph : dict
        Device graph with 'edges' key containing list of [i, j] pairs.
    n : int
        Size of the adjacency matrix (number of qubits in the topology).

    Returns
    -------
    adj : np.ndarray
        Symmetric adjacency matrix of shape (n, n).
        adj[i, j] = 1 if (i, j) is in device edges, else 0.
    """
    adj = np.zeros((n, n), dtype=float)
    for edge in device_graph.get('edges', []):
        i, j = edge[0], edge[1]
        if i < n and j < n:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def _shortest_path_distances(adj: np.ndarray) -> np.ndarray:
    """
    Compute all-pairs shortest path distances via BFS.

    Falls back to a simple BFS implementation when networkx is unavailable.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix of shape (n, n).

    Returns
    -------
    dist : np.ndarray
        Distance matrix where dist[i, j] is shortest path length,
        or np.inf if unreachable.
    """
    n = adj.shape[0]

    # Try networkx for efficiency
    try:
        import networkx as nx
        G = nx.from_numpy_array(adj)
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        dist = np.full((n, n), np.inf)
        for i in range(n):
            if i in lengths:
                for j, d in lengths[i].items():
                    dist[i, j] = d
        return dist
    except ImportError:
        pass

    # Fallback: simple BFS for each source
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)

    for source in range(n):
        visited = np.zeros(n, dtype=bool)
        queue = [(source, 0)]
        visited[source] = True
        while queue:
            current, d = queue.pop(0)
            dist[source, current] = d
            for neighbor in range(n):
                if adj[current, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, d + 1))

    return dist


def routing_swap_cost(mask: np.ndarray, device_adj: np.ndarray) -> float:
    """
    Compute a routing/SWAP overhead proxy for a topology mask.

    The cost penalizes edges in the mask that are not native to the device:
    - For each edge (i, j) with mask[i, j] > threshold:
      - If device_adj[i, j] == 1: cost += 0
      - Else: cost += mask[i, j] * (shortest_path_distance - 1)

    This proxy is deterministic and simple: it counts how many extra
    SWAP operations (roughly) would be needed to implement non-native edges.

    Parameters
    ----------
    mask : np.ndarray
        Topology mask of shape (n, n). Values typically in [0, 1].
    device_adj : np.ndarray
        Device adjacency matrix of shape (n, n).

    Returns
    -------
    cost : float
        Scalar cost proxy. 0 if all mask edges are native.
    """
    n = mask.shape[0]
    threshold = 1e-6

    # Compute distances on device graph
    dist = _shortest_path_distances(device_adj)

    cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only (symmetric)
            mask_val = (mask[i, j] + mask[j, i]) / 2.0
            if mask_val > threshold:
                if device_adj[i, j] > 0:
                    # Native edge: no cost
                    pass
                else:
                    # Non-native edge: cost based on distance
                    d = dist[i, j]
                    if np.isinf(d):
                        # Unreachable: use fallback penalty
                        d = 2.0
                    else:
                        d = max(d - 1.0, 0.0)  # Distance - 1 = SWAP overhead
                    cost += mask_val * d

    return float(cost)


def compilation_penalty(
    weights: np.ndarray,
    masks: List[np.ndarray],
    device_graph: dict,
    lambda_compile: float,
) -> float:
    """
    Compute weighted average routing cost across topologies.

    penalty = lambda_compile * sum_k w_k * routing_swap_cost(M_k, device_adj)

    Parameters
    ----------
    weights : np.ndarray
        Current mixture weights, shape (K,).
    masks : List[np.ndarray]
        List of K topology masks.
    device_graph : dict
        Device graph dictionary.
    lambda_compile : float
        Compilation penalty strength. If 0, returns 0.

    Returns
    -------
    penalty : float
        Scalar compilation penalty.
    """
    if lambda_compile == 0:
        return 0.0

    n = masks[0].shape[0]
    device_adj = device_graph_to_adjacency(device_graph, n)

    total = 0.0
    for k, mask in enumerate(masks):
        cost_k = routing_swap_cost(mask, device_adj)
        total += weights[k] * cost_k

    return lambda_compile * total


def compilation_penalty_grad(
    weights: np.ndarray,
    masks: List[np.ndarray],
    device_graph: dict,
    lambda_compile: float,
) -> np.ndarray:
    """
    Compute gradient of compilation penalty w.r.t. weights.

    d(penalty)/dw_k = lambda_compile * routing_swap_cost(M_k, device_adj)

    Since the penalty is linear in weights, the gradient is simply
    the per-topology costs scaled by lambda_compile.

    To convert to logit gradients, the caller should apply the softmax
    Jacobian: dL/dz_l = sum_k dL/dw_k * w_k * (delta_{kl} - w_l)

    Parameters
    ----------
    weights : np.ndarray
        Current mixture weights, shape (K,).
    masks : List[np.ndarray]
        List of K topology masks.
    device_graph : dict
        Device graph dictionary.
    lambda_compile : float
        Compilation penalty strength. If 0, returns zeros.

    Returns
    -------
    grad : np.ndarray
        Gradient w.r.t. weights, shape (K,).
    """
    K = len(masks)

    if lambda_compile == 0:
        return np.zeros(K)

    n = masks[0].shape[0]
    device_adj = device_graph_to_adjacency(device_graph, n)

    # Gradient w.r.t. w_k is lambda_compile * cost_k
    grad = np.zeros(K)
    for k, mask in enumerate(masks):
        grad[k] = lambda_compile * routing_swap_cost(mask, device_adj)

    return grad


def compilation_penalty_grad_logits(
    weights: np.ndarray,
    masks: List[np.ndarray],
    device_graph: dict,
    lambda_compile: float,
) -> np.ndarray:
    """
    Compute gradient of compilation penalty w.r.t. logits z.

    This converts the weight gradient to logit gradient using the
    softmax Jacobian.

    Parameters
    ----------
    weights : np.ndarray
        Current mixture weights (softmax output), shape (K,).
    masks : List[np.ndarray]
        List of K topology masks.
    device_graph : dict
        Device graph dictionary.
    lambda_compile : float
        Compilation penalty strength.

    Returns
    -------
    grad_z : np.ndarray
        Gradient w.r.t. logits, shape (K,).
    """
    dL_dw = compilation_penalty_grad(weights, masks, device_graph, lambda_compile)

    # Softmax Jacobian: dL/dz_l = w_l * (dL/dw_l - sum_k dL/dw_k * w_k)
    weighted_sum = np.sum(dL_dw * weights)
    grad_z = weights * (dL_dw - weighted_sum)

    return grad_z
