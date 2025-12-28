"""
Identifiability metrics and diagnostics for topology mixture recovery.

Provides:
- Support recovery metrics (precision, recall, F1)
- Dirichlet MAP prior regularization
- Topology confusion baselines
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_support_metrics(
    w_true: np.ndarray,
    w_learned: np.ndarray,
    tau: float = 0.05,
) -> Dict[str, float]:
    """
    Compute support recovery metrics at threshold tau.

    Parameters
    ----------
    w_true : np.ndarray
        Ground-truth mixture weights, shape (K,).
    w_learned : np.ndarray
        Learned mixture weights, shape (K,).
    tau : float
        Threshold for determining support membership.

    Returns
    -------
    metrics : dict
        - 'support_true': set of indices with w_true > tau
        - 'support_learned': set of indices with w_learned > tau
        - 'precision': TP / (TP + FP), or 1.0 if no predictions
        - 'recall': TP / (TP + FN), or 1.0 if no true support
        - 'f1': harmonic mean of precision and recall
        - 'tau': threshold used
    """
    support_true = set(np.where(w_true > tau)[0])
    support_learned = set(np.where(w_learned > tau)[0])

    # True positives: in both supports
    tp = len(support_true & support_learned)
    # False positives: in learned but not true
    fp = len(support_learned - support_true)
    # False negatives: in true but not learned
    fn = len(support_true - support_learned)

    # Precision: fraction of predicted support that is correct
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    # Recall: fraction of true support that was found
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    # F1: harmonic mean
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'support_true': support_true,
        'support_learned': support_learned,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tau': tau,
    }


def dirichlet_log_prior_grad(
    w: np.ndarray,
    alpha: float = 0.3,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute gradient of negative Dirichlet log-prior w.r.t. weights.

    For Dirichlet(alpha) prior on weights w:
        log p(w) = sum_k (alpha - 1) * log(w_k) + const

    We minimize negative log-prior:
        L_prior = -lambda * sum_k (alpha - 1) * log(w_k + eps)
        dL_prior/dw_k = -lambda * (alpha - 1) / (w_k + eps)

    With alpha < 1 (sparse prior), this pushes small weights toward 0.

    Parameters
    ----------
    w : np.ndarray
        Current mixture weights, shape (K,).
    alpha : float
        Dirichlet concentration. alpha < 1 encourages sparsity.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    dL_dw : np.ndarray
        Gradient w.r.t. weights, shape (K,).
    """
    # dL/dw_k = -(alpha - 1) / (w_k + eps)
    # With alpha < 1, (alpha - 1) < 0, so this is positive for small w
    # which pushes small w even smaller (toward 0)
    return -(alpha - 1.0) / (w + eps)


def convert_weight_grad_to_logit_grad(
    dL_dw: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Convert gradient w.r.t. softmax weights to gradient w.r.t. logits.

    Uses the softmax Jacobian:
        dw_k/dz_l = w_k * (delta_{kl} - w_l)

    So:
        dL/dz_l = sum_k dL/dw_k * w_k * (delta_{kl} - w_l)
                = w_l * dL/dw_l - w_l * sum_k dL/dw_k * w_k
                = w_l * (dL/dw_l - sum_k dL/dw_k * w_k)

    Parameters
    ----------
    dL_dw : np.ndarray
        Gradient w.r.t. weights, shape (K,).
    w : np.ndarray
        Current weights (softmax output), shape (K,).

    Returns
    -------
    dL_dz : np.ndarray
        Gradient w.r.t. logits, shape (K,).
    """
    weighted_sum = np.sum(dL_dw * w)
    return w * (dL_dw - weighted_sum)


def run_topology_baselines(
    X: np.ndarray,
    y: np.ndarray,
    topology_names: List[str],
    n: int,
    epochs: int = 150,
    lr: float = 0.2,
    seed: int = 0,
) -> Dict[str, Dict]:
    """
    Train single-topology baselines to assess topology confusability.

    For each topology, trains a model with that topology fixed (one-hot mixture)
    and records the best achieved accuracy and final loss.

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
        Training epochs per topology.
    lr : float
        Learning rate.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Mapping from topology name to {'acc': float, 'loss': float, 'history': dict}
    """
    from .mixture import train_fixed_topology

    results = {}
    for topo in topology_names:
        final_acc, history = train_fixed_topology(
            X, y,
            topology_name=topo,
            n=n,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
        results[topo] = {
            'acc': final_acc,
            'loss': history['loss'][-1],
            'best_acc': max(history['acc']),
            'history': history,
        }

    return results


def format_support_report(
    w_true: np.ndarray,
    w_learned: np.ndarray,
    topology_names: List[str],
    tau: float = 0.05,
) -> str:
    """
    Format a human-readable support recovery report.

    Parameters
    ----------
    w_true : np.ndarray
        Ground-truth weights.
    w_learned : np.ndarray
        Learned weights.
    topology_names : List[str]
        Names of topologies.
    tau : float
        Support threshold.

    Returns
    -------
    report : str
        Formatted report string.
    """
    metrics = compute_support_metrics(w_true, w_learned, tau)

    lines = [
        f"Support Recovery (tau={tau}):",
        f"  True support:    {{{', '.join(topology_names[k] for k in sorted(metrics['support_true']))}}}",
        f"  Learned support: {{{', '.join(topology_names[k] for k in sorted(metrics['support_learned']))}}}",
        f"  Precision: {metrics['precision']:.3f}",
        f"  Recall:    {metrics['recall']:.3f}",
        f"  F1:        {metrics['f1']:.3f}",
        "",
        "Weight comparison:",
    ]

    for k, topo in enumerate(topology_names):
        true_val = w_true[k]
        learned_val = w_learned[k]
        diff = learned_val - true_val
        in_true = "+" if k in metrics['support_true'] else "-"
        in_learned = "+" if k in metrics['support_learned'] else "-"
        lines.append(
            f"  {topo:10}: true={true_val:.4f} learned={learned_val:.4f} "
            f"(diff={diff:+.4f}) support:[{in_true}/{in_learned}]"
        )

    return "\n".join(lines)


def save_identifiability_metrics(
    filepath: str,
    w_true: np.ndarray,
    w_learned: np.ndarray,
    topology_names: List[str],
    l1_error: float,
    kl_div: float,
    accuracy: float,
    tau: float = 0.05,
    baseline_results: Dict = None,
):
    """
    Save identifiability metrics to a JSON file.

    Parameters
    ----------
    filepath : str
        Output file path.
    w_true : np.ndarray
        Ground-truth weights.
    w_learned : np.ndarray
        Learned weights.
    topology_names : List[str]
        Topology names.
    l1_error : float
        L1 distance between true and learned weights.
    kl_div : float
        KL divergence.
    accuracy : float
        Training accuracy.
    tau : float
        Support threshold.
    baseline_results : Dict, optional
        Results from topology baselines.
    """
    import json

    support = compute_support_metrics(w_true, w_learned, tau)

    data = {
        'weights': {
            'true': {topo: float(w_true[k]) for k, topo in enumerate(topology_names)},
            'learned': {topo: float(w_learned[k]) for k, topo in enumerate(topology_names)},
        },
        'errors': {
            'l1': float(l1_error),
            'kl_divergence': float(kl_div),
        },
        'support_recovery': {
            'tau': float(tau),
            'true_support': [topology_names[k] for k in sorted(support['support_true'])],
            'learned_support': [topology_names[k] for k in sorted(support['support_learned'])],
            'precision': float(support['precision']),
            'recall': float(support['recall']),
            'f1': float(support['f1']),
        },
        'accuracy': float(accuracy),
    }

    if baseline_results is not None:
        data['baselines'] = {
            topo: {
                'accuracy': float(res['acc']),
                'loss': float(res['loss']),
            }
            for topo, res in baseline_results.items()
        }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
