# Task 6: Slicing — extract a column

## Hint

To extract a column, you need to select all rows and a specific column. The `.i()` method accepts a tuple where `..` means "all elements along this dimension". So `(.., 1)` means "all rows, column index 1". This gives you a 1D tensor with one value per row.

## Solution

```rust
a.i((.., 1))?
```
