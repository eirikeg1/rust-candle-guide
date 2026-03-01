# Task 4: Create a ones tensor

## Hint

`Tensor::ones()` works just like `Tensor::zeros()`, but fills the tensor with ones. It takes the same three arguments: shape, dtype, and device. The twist here is that you need `DType::F64` instead of `F32` — pay attention to the assertion checking the dtype.

## Solution

```rust
Tensor::ones((2, 2), DType::F64, device)?
```
