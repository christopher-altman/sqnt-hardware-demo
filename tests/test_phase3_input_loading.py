"""
Smoke tests for Phase III input file loading.

Verifies that device graphs and noise models can be loaded from
canonical folders (device_graphs/, noise_models/) and with flexible
path resolution (basename only, full path, legacy phase3_inputs/).
"""

import pytest


def test_load_device_graph_canonical_lnn():
    """Test loading LNN device graph from canonical folder."""
    from sqnt_hardware_demo.compilation import load_device_graph

    # Load with basename (should find in device_graphs/)
    dg = load_device_graph("line_n12.json")

    assert "nodes" in dg
    assert "edges" in dg
    assert len(dg["nodes"]) == 12
    assert len(dg["edges"]) == 11  # Linear chain has n-1 edges


def test_load_device_graph_phase3_schema():
    """Test loading device graph with phase3 schema (n_qubits instead of nodes)."""
    from sqnt_hardware_demo.compilation import load_device_graph

    # Load phase3-style JSON (uses n_qubits, not nodes)
    dg = load_device_graph("line_n8.json")

    # Should auto-populate nodes from n_qubits
    assert "nodes" in dg
    assert "edges" in dg
    assert len(dg["nodes"]) == 8


def test_load_device_graph_grid():
    """Test loading grid device graph."""
    from sqnt_hardware_demo.compilation import load_device_graph

    # Load grid (second priority per spec)
    dg = load_device_graph("grid_3x3.json")

    assert "nodes" in dg
    assert "edges" in dg
    assert len(dg["nodes"]) == 9


def test_load_device_graph_heavy_hex():
    """Test loading heavy-hex device graph."""
    from sqnt_hardware_demo.compilation import load_device_graph

    # Load heavy-hex (third priority per spec)
    dg = load_device_graph("heavy_hex_7q.json")

    assert "nodes" in dg
    assert "edges" in dg
    assert len(dg["nodes"]) == 7


def test_resolve_input_path_basename():
    """Test path resolver with basename only."""
    from sqnt_hardware_demo.compilation import resolve_input_path

    # Should find in device_graphs/
    path = resolve_input_path("line_n12.json", "device_graph")
    assert path.endswith("line_n12.json")
    assert "device_graphs" in path


def test_resolve_input_path_legacy_fallback():
    """Test path resolver falls back to phase3_inputs/ for legacy files."""
    from sqnt_hardware_demo.compilation import resolve_input_path

    # This file only exists in phase3_inputs/
    path = resolve_input_path("compilation_target_gate_set.json", "device_graph")
    assert path.endswith("compilation_target_gate_set.json")
    assert "phase3_inputs" in path


def test_resolve_input_path_not_found():
    """Test path resolver raises FileNotFoundError for missing files."""
    from sqnt_hardware_demo.compilation import resolve_input_path

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_input_path("nonexistent_file.json", "device_graph")

    # Error message should list search paths
    assert "Searched:" in str(exc_info.value)


def test_device_graph_to_adjacency():
    """Test converting device graph to adjacency matrix."""
    from sqnt_hardware_demo.compilation import load_device_graph, device_graph_to_adjacency
    import numpy as np

    dg = load_device_graph("line_n8.json")
    n = 8
    adj = device_graph_to_adjacency(dg, n)

    assert adj.shape == (n, n)
    # Linear chain: adjacent nodes should be connected
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0
    # Non-adjacent should not
    assert adj[0, 2] == 0.0


def test_load_noise_model_canonical():
    """Test loading noise model from canonical folder."""
    from sqnt_hardware_demo.compilation import resolve_input_path
    import json

    # Resolve path to noise model
    path = resolve_input_path("none.json", "noise_model")
    assert "noise_models" in path

    # Verify it's valid JSON
    with open(path, 'r') as f:
        data = json.load(f)
    assert "type" in data or "name" in data


def test_all_canonical_device_graphs_loadable():
    """Test that all canonical device graphs can be loaded."""
    from sqnt_hardware_demo.compilation import load_device_graph
    import os
    from pathlib import Path

    # Get repo root
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent.parent
    device_graphs_dir = repo_root / "device_graphs"

    if not device_graphs_dir.exists():
        pytest.skip("device_graphs/ folder not found")

    # Load all JSON files
    for json_file in device_graphs_dir.glob("*.json"):
        dg = load_device_graph(json_file.name)
        assert "nodes" in dg
        assert "edges" in dg
        assert len(dg["nodes"]) > 0
        assert len(dg["edges"]) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
