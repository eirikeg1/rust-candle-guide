# Task 8: Concatenation

## Hint

`Tensor::cat()` is a static method that joins tensors along a given dimension. It takes a slice of tensor references and a dimension index. To stack vertically (adding more rows), concatenate along dimension 0. Both tensors must have the same number of columns.

## Solution

```rust
Tensor::cat(&[&a, &b], 0)?
```
