# Classification: Neural Networks and XOR

## The XOR Problem

XOR (exclusive or) outputs 1 when inputs differ and 0 when they match:

```
(0, 0) → 0    (0, 1) → 1    (1, 0) → 1    (1, 1) → 0
```

Plot these points and you'll see: no single straight line can separate class 0
from class 1. This is the "not linearly separable" problem that famously
stumped early AI (the Perceptron) in the 1960s.

## How Hidden Layers Solve It

A hidden layer transforms the input into a **new representation** where the
classes *are* separable. For XOR, two hidden neurons can learn to map the four
corners into a space where a line can separate them.

Think of it as the network bending the coordinate plane until the classes fall
on opposite sides of a line.

## ReLU Activation

Without non-linear activations, stacking linear layers is equivalent to a
single linear layer (because matrix multiplication is associative). ReLU
introduces non-linearity:

```
ReLU(x) = max(0, x)
```

It's simple, fast, and works well in practice. It lets the network represent
non-linear boundaries between classes.

## From Logits to Probabilities: Softmax

The network's raw output values are called **logits** — they can be any real
number. To interpret them as probabilities, we apply softmax:

```
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

This maps any vector of logits to a probability distribution: all values
between 0 and 1, summing to 1. The class with the highest probability is
the prediction.

## Cross-Entropy Loss

For classification, we use cross-entropy instead of MSE:

```
loss = -(1/n) Σ log(p_correct_class)
```

It measures how well the predicted probability distribution matches the true
labels. When the model is confident and correct, log(p) ≈ 0 (low loss). When
confident and wrong, log(p) → -∞ (high loss).

In practice, cross-entropy and softmax are computed together numerically
(the "log-sum-exp trick") for stability.

## Optimizers: AdamW

Plain gradient descent uses the same learning rate for every parameter.
**AdamW** improves on this:

- It tracks a running average of each gradient (momentum) to smooth noisy
  updates.
- It tracks a running average of squared gradients to adapt the learning rate
  per parameter — parameters with consistently large gradients get smaller
  steps.
- The "W" adds weight decay (L2 regularization) to prevent overfitting.

AdamW is the default choice for most modern neural networks.

## The Pattern

Classification networks follow the same training loop as regression, with two
differences: the loss function is cross-entropy (not MSE), and the output is
a vector of class probabilities (not a single number).
