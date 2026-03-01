# Task 2: Implement Mlp::new()

## Hint

Use the `candle_nn::linear()` function to create each layer. It takes three arguments: input features, output features, and a `VarBuilder`. Use `vb.pp("name")` to give each layer its own parameter namespace — this prevents name collisions. Layer 1 maps 2 inputs to 16 hidden units, and layer 2 maps 16 hidden units to 2 output classes.

## Solution

```rust
let layer1 = linear(2, 16, vb.pp("layer1"))?;
let layer2 = linear(16, 2, vb.pp("layer2"))?;
Ok(Self { layer1, layer2 })
```
