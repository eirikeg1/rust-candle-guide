# Task 3: Create a zeros tensor

## Hint

`Tensor::zeros()` is a constructor that fills a tensor with zeros. It takes three arguments: a shape (as a tuple like `(3, 4)`), a `DType` specifying the element type, and a device. For this task, use `DType::F32`. Don't forget the `?` — it returns a `Result`.

## Solution

```rust
Tensor::zeros((3, 4), DType::F32, device)?
```
