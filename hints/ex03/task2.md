# Task 2: Forward pass

## Hint

The forward pass computes `prediction = x @ weight + bias`. Use `.matmul()` on `x` with the weight tensor. Since `weight` is a `Var`, access its underlying tensor with `.as_tensor()`. Then use `.broadcast_add()` to add the bias — broadcast handles the shape difference between the matmul result (50x1) and the bias (1,).

## Solution

```rust
x.matmul(weight.as_tensor())?.broadcast_add(bias.as_tensor())?
```
