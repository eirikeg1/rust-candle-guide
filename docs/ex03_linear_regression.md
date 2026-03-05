# Linear Regression: Your First ML Model

## Bridging from Tensors to Training: Autodiff Primer

In Exercise 2, you worked with regular `Tensor` values — fixed data that you
manipulated with arithmetic and reshaping. Training a model requires something
new: **parameters that can be updated based on how wrong the model is**. This
is where `Var` and automatic differentiation come in.

### `Var` vs `Tensor`

A `Var` is a `Tensor` that Candle tracks for gradient computation:

```rust
use candle_core::{Device, Tensor, Var};

// Regular tensor — just data, no gradients
let x = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;

// Var — a trainable parameter that tracks gradients
let w = Var::new(&[0.0f32], &Device::Cpu)?;
```

You can use a `Var` anywhere a `Tensor` is expected by calling `.as_tensor()`.
The key difference: when you call `.backward()` on a loss computed from `Var`
values, Candle computes the gradient of the loss with respect to each `Var`.

### What does `.backward()` return?

Calling `loss.backward()` returns a `GradStore` — a map from tensors to their
gradients. You look up a specific parameter's gradient with `.get()`:

```rust
// Minimal example: compute a gradient
let w = Var::new(&[2.0f32], &Device::Cpu)?;
let loss = w.as_tensor().sqr()?.mean_all()?;  // loss = w^2 = 4.0
let grads = loss.backward()?;
let grad_w = grads.get(w.as_tensor()).unwrap();
// grad_w = 2*w = 4.0 (the derivative of w^2)
```

This is **automatic differentiation** — Candle recorded that `loss` was computed
by squaring `w`, so it knows the gradient is `2*w`. No matter how complex your
forward computation is, `.backward()` computes exact gradients automatically.

### From gradients to learning

Once you have gradients, a single parameter update looks like:

```rust
w.set(&(w.as_tensor() - (grad_w * learning_rate)?)?)?;
```

This is gradient descent: move each parameter a small step in the direction that
reduces the loss. Repeat this in a loop, and the model learns. That's exactly
what you'll implement in this exercise.

---

## The Model

Linear regression fits a straight line to data:

```
y = w·x + b
```

- **w** (weight) is the slope — how much y changes per unit of x.
- **b** (bias) is the y-intercept — the baseline when x = 0.

With one input feature, this is a line. With multiple features, it's a
hyperplane. Either way, the idea is the same: find the w and b that best
explain the data.

## The Loss Function: Mean Squared Error

How do we measure "best"? We define a **loss function** that quantifies how
wrong the model is. For regression, MSE is standard:

```
MSE = (1/n) Σ (predicted - actual)²
```

Why squaring?
- It makes all errors positive (overshooting and undershooting both count).
- It penalizes large errors more than small ones (quadratic growth).
- It's differentiable everywhere, which we need for gradient descent.

## Gradient Descent

Gradient descent is how we minimize the loss:

1. **Compute the gradient** — the partial derivative of the loss with respect
   to each parameter. The gradient points in the direction of steepest
   *increase*.

2. **Step opposite** — update each parameter by subtracting a fraction of its
   gradient:
   ```
   w ← w - lr · ∂loss/∂w
   b ← b - lr · ∂loss/∂b
   ```

3. **Repeat** — the loss decreases each step (if the learning rate is small
   enough), and the parameters converge toward optimal values.

## Automatic Differentiation

Computing gradients by hand is tedious and error-prone. ML frameworks use
**automatic differentiation** (autodiff): they record every operation during
the forward pass, then replay them backward to compute exact gradients.

In Candle, this is the `backward()` method. It returns a `GradStore` mapping
each `Var` to its gradient tensor.

## The Training Loop Pattern

Almost every ML training follows this pattern:

```
for each epoch:
    prediction = model(input)           # forward pass
    loss = loss_function(prediction, target)
    gradients = loss.backward()          # backward pass
    update parameters using gradients    # optimizer step
```

This loop is the same whether you're training a linear regression or GPT-4.
The difference is the model's complexity and the optimizer's sophistication.

## Learning Rate

The learning rate (lr) controls step size:
- **Too large** → the model overshoots and diverges (loss explodes).
- **Too small** → convergence is painfully slow.
- **Just right** → loss decreases smoothly to a low value.

For simple linear regression, values like 0.01–0.1 work well. More complex
models need adaptive optimizers (like AdamW) that tune the rate automatically.

## Convergence

A well-tuned model's loss curve drops steeply at first (large errors, large
gradients) then flattens out (approaching the optimum). If your loss plateaus
at a high value, the model may lack capacity. If it oscillates, the learning
rate is too high.
