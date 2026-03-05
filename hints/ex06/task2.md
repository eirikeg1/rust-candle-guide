# Task 2: Implement forward pass

## Hint

The forward pass for `IrisNet` follows the same pattern as Exercise 4 and 5: chain linear layers with ReLU activations between them. Apply activation after each hidden layer, but **not** after the final layer (cross-entropy loss handles softmax internally).

The pattern is: input → layer1 → relu → layer2 → relu → layer3.

## Solution

```rust
let x = self.layer1.forward(xs)?.relu()?;
let x = self.layer2.forward(&x)?.relu()?;
self.layer3.forward(&x)
```
