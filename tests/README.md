# SQNT Hardware Demo Tests

This directory contains comprehensive unit tests for the SQNT hardware demo project.

## Running Tests

To run all tests:

```bash
# Using the virtual environment
.venv/bin/python -m pytest tests/test_sqnt_hardware_demo.py -v

# Or with activated virtual environment
source .venv/bin/activate
pytest tests/test_sqnt_hardware_demo.py -v
```

## Test Coverage

### 1. TestMakeGraphMask (11 tests)
Tests for `make_graph_mask` function covering:
- **Graph types**: chain, ring, star, complete
- **Self-loops**: with and without `include_self`
- **Normalization**: with and without `normalize`
- **Edge cases**: invalid graph types, case-insensitivity

### 2. TestSQNTLayer (13 tests)
Tests for `SQNTLayer` class covering:
- **Initialization**: proper shape and seed determinism
- **Forward pass**: output shapes, ranges, specific masks (identity, zero), determinism
- **Loss and gradients**: shapes, non-negative loss, finite gradients, mask respect
- **Parameter updates**: step method correctness

### 3. TestTrainForAlpha (8 tests)
Tests for `train_for_alpha` function covering:
- **Return values**: valid accuracy range [0, 1]
- **Alpha values**: edge cases (0, 1) and intermediate values
- **Topologies**: different topology combinations
- **Determinism**: same seed produces same results
- **Training efficacy**: improves over random baseline
- **Hyperparameters**: different learning rates

### 4. TestSweepAlphas (9 tests)
Tests for `sweep_alphas` function covering:
- **Output format**: correct shape and type
- **Validation**: all values in valid range
- **Edge cases**: single alpha, empty array
- **Determinism**: reproducible results
- **Orchestration**: correct aggregation and ordering
- **Topologies**: different topology pairs

### 5. TestSigmoid (4 tests)
Tests for `sigmoid` helper function covering:
- **Special values**: zero input
- **Saturation**: large positive/negative values
- **Range**: output always in (0, 1)

### 6. TestMakeSynthetic (3 tests)
Tests for `make_synthetic` helper function covering:
- **Output shapes**: correct dimensions
- **Determinism**: same seed produces same data
- **Label validity**: binary labels (0 or 1)

## Total: 46 Tests

All tests pass successfully and provide comprehensive coverage of the core functionality.
