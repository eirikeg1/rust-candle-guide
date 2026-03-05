# Task 1: Understand the MLP struct

## Hint

The `Mlp` struct is already defined for you with two fields: `layer1` and `layer2`, both of type `Linear`. A `Linear` layer stores a weight matrix and a bias vector. It performs the operation `output = input @ weight^T + bias`. This is the same linear transformation from Exercise 3, but packaged into a reusable module.

The struct definition tells you the architecture: two linear layers stacked. Your job in the next tasks is to initialize these layers (in `new()`) and wire them together (in `forward()`).

## Key Concepts

```rust
// Linear is a candle_nn type that wraps weight + bias
struct Mlp {
    layer1: Linear,  // maps input features → hidden units
    layer2: Linear,  // maps hidden units → output classes
}
```

No code to write here — just make sure you understand the struct before implementing `new()` and `forward()`.
