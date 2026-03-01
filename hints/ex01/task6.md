# Task 6: Create a random uniform tensor

## Hint

`Tensor::rand()` generates values from a uniform distribution. It takes four arguments: low bound, high bound, shape, and device. Use `0f32` for the low end and `1.0` for the high end to get values in the [0, 1) range. The first argument needs the `f32` suffix so Rust knows the type.

## Solution

```rust
Tensor::rand(0f32, 1.0, (2, 5), device)?
```
