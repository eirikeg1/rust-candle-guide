# Task 4: Training step

## Hint

Same three-step pattern as Exercise 4, but with MSE loss instead of cross-entropy (since this is regression, not classification):

1. **Forward pass:** Call `model.forward(&x_train)?` to get predictions.
2. **MSE loss:** Compute `(pred - y_train).sqr().mean_all()` — subtract, square, and average.
3. **Optimize:** Call `optimizer.backward_step(&loss)?`.

Remember to delete the placeholder `let loss = ...` line below the todo once you've defined `loss` yourself.

## Solution

```rust
let pred = model.forward(&x_train)?;
let loss = (&pred - &y_train)?.sqr()?.mean_all()?;
optimizer.backward_step(&loss)?;
```
