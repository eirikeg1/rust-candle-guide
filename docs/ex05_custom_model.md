# Deeper Networks: Function Approximation

## The Universal Approximation Theorem

A neural network with a single hidden layer (and enough neurons) can
approximate *any* continuous function to arbitrary precision. This is the
**universal approximation theorem** — it's why neural networks are so powerful.

In practice, however, "enough neurons" can be astronomically large. Deeper
networks with multiple smaller layers tend to learn complex functions more
efficiently than a single wide layer.

## Why Deeper > Wider (In Practice)

Each layer in a network builds increasingly abstract representations:

- Layer 1 might learn simple features (edges in images, basic patterns in data)
- Layer 2 combines those into mid-level features (shapes, curves)
- Layer 3 composes those into high-level concepts (objects, relationships)

A 3-layer network with 32 neurons per layer has 32 + 32·32 + 32 ≈ 1,100
parameters, but can represent functions that a single layer of 1,100 neurons
cannot efficiently learn.

## GELU Activation

GELU (Gaussian Error Linear Unit) is a smoother alternative to ReLU:

```
GELU(x) ≈ x · Φ(x)
```

where Φ is the Gaussian CDF. Unlike ReLU which has a hard cutoff at 0, GELU
smoothly transitions, allowing small negative values through. This helps with
gradient flow during training and often produces slightly better results for
function approximation and language models.

## Function Approximation as a Test Case

Fitting `y = sin(x)` is a useful test because:

- The true function is known, so you can measure exact error.
- It requires non-linearity (a linear model can't curve).
- It's smooth, so a well-trained network should approximate it closely.
- You can test generalization by evaluating on unseen x values.

## Overfitting vs. Generalization

- **Overfitting**: the model memorizes training data (low training loss) but
  performs poorly on new data.
- **Generalization**: the model captures the underlying pattern and works well
  on unseen inputs.

For sin(x), overfitting means the network perfectly hits training points but
produces garbage between them. Generalization means it learns the smooth sine
curve. With clean synthetic data and a reasonably sized network, overfitting
is unlikely, but it becomes a major concern with real-world data.

## The Module Trait Pattern

In Candle, the `Module` trait provides a standard interface for models:

```rust
trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

This pattern separates model definition (the `new()` constructor creates
layers) from computation (the `forward()` method chains them). It allows
models to be composed — a `Module` can contain other `Module`s as fields,
enabling complex architectures from simple building blocks.
