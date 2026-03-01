# Task 2: Create a 2D tensor

## Hint

`Tensor::new()` also works with nested arrays. Pass a reference to a 2D array (an array of arrays) to create a 2D tensor. The shape is inferred from the array dimensions — a `&[[..], [..]]` gives you a 2-row tensor, and the inner arrays determine the number of columns. Use `f32` suffix on the first element.

## Solution

```rust
Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], device)?
```
