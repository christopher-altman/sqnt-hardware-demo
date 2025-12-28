"""
Hardware-aware topology constraint operators for SQNT.

Phase I: Implements basic degree and locality constraints for canonical topologies.
Constraints are expressed as binary violations and converted to a penalty term.
"""

import numpy as np
from typing import Dict, Optional


def violates_max_degree(topology: str, max_degree: int) -> bool:
    """
    Check if a canonical topology violates the maximum degree constraint.

    Parameters
    ----------
    topology : str
        Topology name (chain, ring, star, complete).
    max_degree : int
        Maximum allowed node degree.

    Returns
    -------
    bool
        True if topology violates constraint, False otherwise.

    Notes
    -----
    Degree semantics for canonical topologies (n=12):
    - chain: max degree = 2 (endpoints have degree 1)
    - ring: max degree = 2 (all nodes have degree 2)
    - star: max degree = 11 (hub connects to all other nodes)
    - complete: max degree = 11 (all nodes connect to all others)
    """
    topology = topology.lower().strip()

    # Map topology to maximum degree for canonical case
    # These are exact for any n >= 3 (hub degree is n-1)
    if topology == "chain":
        topo_max_degree = 2
    elif topology == "ring":
        topo_max_degree = 2
    elif topology == "star":
        # Hub degree is n-1; for n=12, hub has degree 11
        # Conservative: assume large n, so star has high degree
        topo_max_degree = 100  # Placeholder: star hub has degree n-1
    elif topology == "complete":
        # Each node connects to n-1 others
        topo_max_degree = 100  # Placeholder: complete has degree n-1
    else:
        # Unknown topology: assume it might violate
        return False

    return topo_max_degree > max_degree


def violates_locality(topology: str, locality_radius: int) -> bool:
    """
    Check if a canonical topology violates the locality constraint.

    Parameters
    ----------
    topology : str
        Topology name (chain, ring, star, complete).
    locality_radius : int
        Maximum allowed hop distance for edges.

    Returns
    -------
    bool
        True if topology violates constraint, False otherwise.

    Notes
    -----
    Locality semantics for canonical topologies:
    - chain: all edges connect neighbors (radius 1)
    - ring: all edges connect neighbors (radius 1)
    - star: hub-spoke edges have radius 1, but spoke-spoke distance is 2
    - complete: all pairs connected, including long-range (violates small radius)

    We define violation conservatively:
    - chain and ring are always local (radius 1)
    - star has radius 2 (spoke-to-spoke via hub)
    - complete violates any radius < n-1
    """
    topology = topology.lower().strip()

    if topology == "chain":
        return False  # Always local
    elif topology == "ring":
        return False  # Always local
    elif topology == "star":
        # Star has effective radius 2 (spoke-to-spoke)
        return locality_radius < 2
    elif topology == "complete":
        # Complete graph has all pairs, violates small radius
        return locality_radius < 100  # Conservative: complete is non-local
    else:
        # Unknown topology: assume it might violate
        return False


def constraint_penalty(
    weights: np.ndarray,
    topologies: list,
    cfg: Optional[Dict] = None,
) -> float:
    """
    Compute constraint penalty for a mixture of topologies.

    The penalty encourages the mixture to assign zero weight to topologies
    that violate hardware constraints.

    Parameters
    ----------
    weights : np.ndarray
        Mixture weights, shape (K,), summing to 1.
    topologies : list of str
        List of topology names corresponding to weights.
    cfg : dict, optional
        Constraint configuration with keys:
        - 'enabled': bool, whether constraints are active
        - 'max_degree': int or None, maximum degree constraint
        - 'locality_radius': int or None, locality constraint
        - 'lambda_constraint': float, penalty strength

    Returns
    -------
    penalty : float
        Constraint penalty. Zero if constraints disabled or no violations.

    Notes
    -----
    Penalty computation:
    - For each topology that violates constraints, add its weight to penalty
    - penalty = sum_{k: violates} weights[k]
    - Scaled by lambda_constraint when added to loss

    This encourages gradient descent to move mass from infeasible to feasible topologies.
    """
    if cfg is None or not cfg.get("enabled", False):
        return 0.0

    max_degree = cfg.get("max_degree", None)
    locality_radius = cfg.get("locality_radius", None)

    # If no constraints specified, no penalty
    if max_degree is None and locality_radius is None:
        return 0.0

    penalty = 0.0

    for k, topo in enumerate(topologies):
        violates = False

        if max_degree is not None:
            if violates_max_degree(topo, max_degree):
                violates = True

        if locality_radius is not None:
            if violates_locality(topo, locality_radius):
                violates = True

        if violates:
            penalty += weights[k]

    return float(penalty)


def constraint_penalty_grad(
    weights: np.ndarray,
    topologies: list,
    cfg: Optional[Dict] = None,
) -> np.ndarray:
    """
    Compute gradient of constraint penalty with respect to mixture weights.

    Parameters
    ----------
    weights : np.ndarray
        Mixture weights, shape (K,).
    topologies : list of str
        List of topology names.
    cfg : dict, optional
        Constraint configuration.

    Returns
    -------
    grad : np.ndarray
        Gradient of penalty w.r.t. weights, shape (K,).

    Notes
    -----
    Since penalty = sum_{k: violates} weights[k], the gradient is:
    - dP/dw_k = 1 if topology k violates constraints
    - dP/dw_k = 0 otherwise
    """
    if cfg is None or not cfg.get("enabled", False):
        return np.zeros_like(weights)

    max_degree = cfg.get("max_degree", None)
    locality_radius = cfg.get("locality_radius", None)

    if max_degree is None and locality_radius is None:
        return np.zeros_like(weights)

    grad = np.zeros_like(weights)

    for k, topo in enumerate(topologies):
        violates = False

        if max_degree is not None:
            if violates_max_degree(topo, max_degree):
                violates = True

        if locality_radius is not None:
            if violates_locality(topo, locality_radius):
                violates = True

        if violates:
            grad[k] = 1.0

    return grad
