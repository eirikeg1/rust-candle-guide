# Linear Regression: Your First ML Model

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
