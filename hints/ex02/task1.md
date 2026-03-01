# Task 1: Element-wise addition

## Hint

Candle overloads the `+` operator for tensors. You can add two tensors element-wise using `&a + &b` (note the references). The result is a `Result<Tensor>`, so you need the `?` operator. Both tensors must have the same shape.

## Solution

```rust
(&a + &b)?
```
