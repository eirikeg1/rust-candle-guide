# Task 5: Slicing — extract first row

## Hint

The `.i()` method indexes into a tensor. To extract a single row, pass the row index as an integer. For the first row, that's `0`. This reduces the dimensionality — a row from a 2D tensor becomes a 1D tensor. Returns a `Result`.

## Solution

```rust
a.i(0)?
```
