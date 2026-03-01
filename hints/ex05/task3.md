# Task 3: Implement forward

## Hint

Chain three layers with activations between them: layer1 → activation → layer2 → activation → layer3. Use `.forward()` on each Linear layer and `.apply(&self.activation)` to apply the activation function. The `Activation` enum implements `Module`, so `.apply()` works on it. No activation after the final layer — the output should be a raw value.

## Solution

```rust
let x = self.layer1.forward(xs)?.apply(&self.activation)?;
let x = self.layer2.forward(&x)?.apply(&self.activation)?;
self.layer3.forward(&x)
```
