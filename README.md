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

Each exercise module (`src/exercises/ex*.rs`) contains `todo!()` placeholders for you to fill in. The binaries (`src/bin/ex*.rs`) run the exercises and check results. Hints are available in the `hints/` directory.

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

Select an exercise from the sidebar and click **Run** to start. The GUI shows live training metrics: loss curves for all exercises, weight/bias convergence for linear regression, decision boundaries for classification, and confusion matrices for multi-class problems.

## Dependencies

- [candle-core](https://crates.io/crates/candle-core) / [candle-nn](https://crates.io/crates/candle-nn) 0.9 -- tensor library and neural network modules
- [eframe](https://crates.io/crates/eframe) 0.33 / [egui_plot](https://crates.io/crates/egui_plot) 0.34 -- GUI and plotting (only used by the `gui` binary)

## Requirements

- Rust 1.85+ (this project uses edition 2024)

## What's Next

After completing these exercises, some directions to explore:

- **Convolutional neural networks (CNNs)** — image classification with Candle's `candle_nn::conv2d`
- **Loading pretrained models** — use [HuggingFace Hub](https://huggingface.co/docs/candle) to load weights from published models
- **GPU acceleration** — swap `Device::Cpu` for `Device::cuda_if_available()` to run on NVIDIA GPUs
