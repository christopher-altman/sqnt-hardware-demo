"""sqnt-hardware-demo: superposed topology spatialization demo."""

from .graphs import make_graph_mask
from .sqnt_layer import SQNTLayer
from .mixture import TopologyMixture, train_learned_mixture, train_fixed_topology, train_random_mixture
from .experiments import (
    sample_ground_truth_mixture,
    generate_planted_mixture_data,
    train_mixture_recovery,
    run_recovery_phase_diagram,
)
from .mlp import SQNTMLP, train_mlp_mixture
from .identifiability import (
    compute_support_metrics,
    run_topology_baselines,
    format_support_report,
    save_identifiability_metrics,
)
