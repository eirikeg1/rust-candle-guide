# Task 4: Reshape

## Hint

The `.reshape()` method changes a tensor's shape without changing its data. Pass the new shape as a tuple. The total number of elements must stay the same — a (2, 3) tensor has 6 elements, so (3, 2) works. Data stays in row-major order, so the values "reflow" into the new shape.

## Solution

```rust
a.reshape((3, 2))?
```
