# ExecuTorch ANE Benchmark

Benchmark test for measuring ExecuTorch model execution performance on Apple Neural Engine (ANE).

## Setup

### 1. Clone the repository

```bash
git clone -b ane-repro https://github.com/metascroy/executorch-testing.git
cd executorch-testing
git submodule update --init --recursive
```

### 2. Install ExecuTorch

```bash
pushd third-party/executorch
python install_executorch.py --editable
popd
```

### 3. Build the test

```bash
./build_tests.sh
```

This will build the `test_model` binary in `cmake-out/`.

## Running the benchmark

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_PATH` | Yes | - | Path to the `.pte` model file |
| `DELAY_MS` | No | 0 | Delay in milliseconds between iterations |
| `EXCLUDE_N` | No | 0 | Number of initial iterations to exclude from statistics |

### Case 1: Tight loop (no delay)

```bash
EXCLUDE_N=1 DELAY_MS=0 MODEL_PATH=/path/to/xl_model_dummy.pte ./cmake-out/test_model
```

Output:
```
Iterations: 300
Delay between iterations: 0 ms
Exclude first N from stats: 1
Running warmup...
Running 300 iterations...

=== Results (excluding first 1) ===
Iterations used for stats: 299
Total wall time: 747.896 ms
Total execution time (forward only): 744.059 ms
Average per iteration: 2.48849 ms
Min iteration time: 2.41162 ms
Max iteration time: 3.26883 ms
```

### Case 2: With 10ms delay between iterations

```bash
EXCLUDE_N=1 DELAY_MS=10 MODEL_PATH=/path/to/xl_model_dummy.pte ./cmake-out/test_model
```

Output:
```
Iterations: 300
Delay between iterations: 10 ms
Exclude first N from stats: 1
Running warmup...
Running 300 iterations...

=== Results (excluding first 1) ===
Iterations used for stats: 299
Total wall time: 5278.14 ms
Total execution time (forward only): 1688.66 ms
Average per iteration: 5.64768 ms
Min iteration time: 2.62988 ms
Max iteration time: 10.0477 ms
Delay overhead: 3589.49 ms
```

### Key Finding

**ANE execution time is 2.3x slower with a 10ms delay between predictions.**

| Scenario | Avg per iteration |
|----------|-------------------|
| Tight loop (DELAY_MS=0) | 2.49 ms |
| With delay (DELAY_MS=10) | 5.65 ms |
| **Slowdown** | **2.27x** |

This demonstrates that the Apple Neural Engine performs significantly better when running predictions in a tight loop, possibly due to the ANE staying in an active/warm state.
