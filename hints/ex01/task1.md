# Task 1: Create a 1D tensor

## Hint

You need `Tensor::new()` which takes two arguments: a reference to your data and a device. For a 1D tensor, pass a slice of `f32` values. Use the `f32` suffix on the first element (e.g., `1.0f32`) so Rust infers the correct type for the whole array. The function returns a `Result`, so use `?` to unwrap it.

## Solution

```rust
Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], device)?
```
