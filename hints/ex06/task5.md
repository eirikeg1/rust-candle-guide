# Task 5: Training step

## Hint

Identical pattern to Exercise 4's training step — this is classification, so use cross-entropy loss:

1. **Forward pass:** Call `model.forward(&x_train)?` to get logits.
2. **Loss:** Use `loss::cross_entropy(&logits, &y_train)?` with raw logits and integer labels.
3. **Optimize:** Call `optimizer.backward_step(&loss)?`.

Remember to delete the placeholder `let loss = ...` line below the todo once you've defined `loss` yourself.

## Solution

```rust
let logits = model.forward(&x_train)?;
let loss = loss::cross_entropy(&logits, &y_train)?;
optimizer.backward_step(&loss)?;
```
