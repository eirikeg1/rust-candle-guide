# Task 5: Create a random normal tensor

## Hint

`Tensor::randn()` generates values from a normal (Gaussian) distribution. It takes four arguments: mean, standard deviation, shape, and device. The mean and std need to be concrete float types — use `0f32` for mean and `1.0` for std to get the standard normal distribution. The shape is a tuple like `(3, 3)`.

## Solution

```rust
Tensor::randn(0f32, 1.0, (3, 3), device)?
```
