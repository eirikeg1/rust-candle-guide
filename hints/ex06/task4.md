# Task 4: Implement forward

## Hint

Chain three layers with ReLU activations: layer1 → relu → layer2 → relu → layer3. Use `.forward()` on each Linear layer and `.relu()` on the result for activation. No activation after the final layer — cross-entropy loss expects raw logits. This is the same pattern as Exercise 4, just with an extra hidden layer.

## Solution

```rust
let x = self.layer1.forward(xs)?.relu()?;
let x = self.layer2.forward(&x)?.relu()?;
self.layer3.forward(&x)
```
