# Task 2: Element-wise multiplication

## Hint

Just like addition, Candle overloads the `*` operator for element-wise multiplication. Use `&a * &b` with references to both tensors. The result is a `Result<Tensor>`, so append `?`. The shapes must match.

## Solution

```rust
(&a * &b)?
```
