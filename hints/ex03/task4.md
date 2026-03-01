# Task 4: Backward pass and parameter update

## Hint

Three steps:

1. **Compute gradients:** Call `loss.backward()` which returns a `GradStore` — a map from tensors to their gradients.
2. **Extract gradients:** Use `grads.get(var.as_tensor())` to look up the gradient for each `Var`. This returns a `Result`.
3. **Update parameters:** For each Var, compute `new_value = old_value - learning_rate * gradient` and apply it with `var.set(&new_value)?`. Use tensor arithmetic with the learning rate as an `f64` scalar.

## Solution

```rust
let grads = loss.backward()?;
let grad_w = grads.get(weight.as_tensor()).unwrap();
let grad_b = grads.get(bias.as_tensor()).unwrap();
weight.set(&(weight.as_tensor() - (learning_rate * grad_w)?)?)?;
bias.set(&(bias.as_tensor() - (learning_rate * grad_b)?)?)?;
```
