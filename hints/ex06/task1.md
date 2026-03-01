# Task 1: Load the Iris dataset

## Hint

The utility library provides `iris::load_iris_split()` which loads the Iris dataset and splits it into training and test sets. It takes a `&Device` and returns a `Result` containing a tuple of four tensors: `(x_train, y_train, x_test, y_test)`. Just call it and unpack the result.

## Solution

```rust
iris::load_iris_split(device)?
```
