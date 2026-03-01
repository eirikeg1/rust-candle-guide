# Task 1: Create trainable parameters

## Hint

A `Var` wraps a tensor and tracks gradients. Use `Var::from_tensor()` to create one from an existing tensor. First create a zeros tensor with the right shape and dtype (`DType::F32`), then wrap it in a `Var`. The weight needs shape `(1, 1)` for matrix multiplication with x, and the bias needs shape `(1,)` as a 1D vector.

## Solution

```rust
// weight
Var::from_tensor(&Tensor::zeros((1, 1), DType::F32, device)?)?

// bias
Var::from_tensor(&Tensor::zeros((1,), DType::F32, device)?)?
```
