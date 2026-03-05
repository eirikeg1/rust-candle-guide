# Task 1: Understand the SineNet struct

## Hint

The `SineNet` struct has three linear layers and an activation function. This is a deeper network than Exercise 4's two-layer MLP. The architecture is 1 → 32 → 32 → 1:

- **layer1**: maps 1 input (x value) to 32 hidden units
- **layer2**: maps 32 hidden units to 32 hidden units
- **layer3**: maps 32 hidden units to 1 output (predicted y)
- **activation**: `Activation::Gelu` — a smooth alternative to ReLU

The extra depth helps the network learn the curves of sin(x). With only one hidden layer, you'd need many more neurons to get a good fit.

## Key Concepts

```rust
struct SineNet {
    layer1: Linear,          // 1 → 32
    layer2: Linear,          // 32 → 32
    layer3: Linear,          // 32 → 1
    activation: Activation,  // GELU between layers
}
```

No code to write here — review the struct before implementing `new()` and `forward()`.
