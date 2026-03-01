# Task 3: Matrix multiplication

## Hint

Use the `.matmul()` method on a tensor to perform matrix multiplication. Call `a.matmul(&c)?` — the inner dimensions must match (a is 2x3, c is 3x2, so the 3s align). The result shape is the outer dimensions: 2x2. Returns a `Result`, so use `?`.

## Solution

```rust
a.matmul(&c)?
```
