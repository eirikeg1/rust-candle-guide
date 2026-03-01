# Rust ML Learning Exercises

Progressive machine learning exercises in Rust using the [Candle](https://github.com/huggingface/candle) framework. Each exercise introduces a core ML concept, from tensor basics to neural network classification.

## Exercises

| # | Topic | Concept |
|---|-------|---------|
| 1 | Tensor Basics | Creating tensors (from data, zeros, ones, random) |
| 2 | Tensor Operations | Arithmetic, matmul, reshape, slicing, reductions |
| 3 | Linear Regression | Manual gradient descent with `Var` and `backward()` |
| 4 | XOR Classification | MLP with cross-entropy loss and AdamW |
| 5 | Sine Approximation | Deeper network with GELU activations |
| 6 | Iris Classification | Multi-class classification with train/test evaluation |

Each exercise binary (`src/bin/ex*.rs`) contains `todo!()` placeholders for you to fill in. Hints are available in the `hints/` directory.

## Running

### CLI exercises

```sh
cargo run --bin ex01_tensor_basics
cargo run --bin ex02_tensor_ops
# ... through ex06_iris
```

### GUI

An interactive desktop GUI lets you run all exercises with live training visualization (loss curves, scatter plots, decision boundaries, confusion matrices).

```sh
cargo run --bin gui
```

Select an exercise from the sidebar and click **Run** to start.

## Dependencies

- [candle-core](https://crates.io/crates/candle-core) / [candle-nn](https://crates.io/crates/candle-nn) 0.9 -- tensor library and neural network modules
- [eframe](https://crates.io/crates/eframe) 0.33 / [egui_plot](https://crates.io/crates/egui_plot) 0.34 -- GUI and plotting (only used by the `gui` binary)
