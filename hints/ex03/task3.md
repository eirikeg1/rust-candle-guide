# Task 3: MSE loss

## Hint

Mean Squared Error is computed in three steps: (1) subtract prediction from target to get the error, (2) square each error with `.sqr()`, (3) take the mean of all squared errors with `.mean_all()`. Use the `&` reference operator on tensors when subtracting, and chain the `?` operator after each fallible call.

## Solution

```rust
(&pred - &y)?.sqr()?.mean_all()?
```
