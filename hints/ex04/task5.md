# Task 5: Training step

## Hint

Three steps each iteration:

1. **Forward pass:** Call `model.forward(&x_train)?` to get logits (raw predictions).
2. **Compute loss:** Use `loss::cross_entropy(&logits, &y_train)?` — it takes the raw logits and integer class labels.
3. **Optimize:** Call `optimizer.backward_step(&loss)?` — this computes gradients and updates all parameters in one call. Much simpler than manual gradient descent!

Remember to delete the placeholder `let loss = ...` line below the todo once you've defined `loss` yourself.

## Solution

```rust
let logits = model.forward(&x_train)?;
let loss = loss::cross_entropy(&logits, &y_train)?;
optimizer.backward_step(&loss)?;
```
