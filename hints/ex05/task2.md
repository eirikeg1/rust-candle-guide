# Task 2: Implement SineNet::new()

## Hint

Create three linear layers using `candle_nn::linear()`, each with its own namespace via `vb.pp("name")`. The architecture is 1 → 32 → 32 → 1 (one input feature, two hidden layers of 32 units, one output). Store `Activation::Gelu` as the activation — it's an enum variant, not a function call.

## Solution

```rust
let layer1 = linear(1, 32, vb.pp("layer1"))?;
let layer2 = linear(32, 32, vb.pp("layer2"))?;
let layer3 = linear(32, 1, vb.pp("layer3"))?;
Ok(Self {
    layer1,
    layer2,
    layer3,
    activation: Activation::Gelu,
})
```
