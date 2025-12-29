# Phase III Concrete Inputs (Drop-in Templates)

This folder contains **minimal, concrete JSON inputs** for Phase III work:
- **Device graphs**: toy-heavy-hex, 3x3 grid, and LNN chain
- **Compilation target**: common gate-set (CX + {RZ,SX})
- **Routing cost weights**: default scoring weights
- **Noise models**: simple channel-level templates (not calibrated)

These are intended for:
1) unit tests (parsing + scoring)
2) example CLI invocations
3) deterministic simulation harnesses

## Files

### Device graphs
- `device_heavy_hex_min.json` — toy heavy-hex-like sparse connectivity (7 qubits)
- `device_grid_3x3.json` — 3x3 nearest-neighbor grid (9 qubits)
- `device_lnn_8.json` — linear nearest-neighbor chain (8 qubits)

### Compilation targets
- `compilation_target_gate_set.json`

### Routing costs
- `routing_cost_weights.json`

### Noise models (templates)
- `noise_model_minimal.json`
- `noise_model_amplitude_damping.json`

## Notes
- The heavy-hex JSON is **illustrative** (not an exact IBM device coupling map).
- Noise models are **templates** (probabilities per gate); convert to your simulator’s format as needed.
