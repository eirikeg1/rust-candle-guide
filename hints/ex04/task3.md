# Task 3: Implement forward

## Hint

The forward pass chains layers and activations. Call `.forward()` on each `Linear` layer (from the `Module` trait), and `.relu()` on the tensor for activation. The pattern is: input → layer1 → relu → layer2. No activation after the last layer — the cross-entropy loss function handles softmax internally.

## Solution

```rust
let x = self.layer1.forward(xs)?.relu()?;
self.layer2.forward(&x)
```
