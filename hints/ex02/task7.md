# Task 7: Reductions

## Hint

This task has two parts:

**(a) Mean of all elements:** Use `.mean_all()` which reduces every element to a single scalar value. No arguments needed — it averages everything.

**(b) Sum along dimension 1:** Use `.sum()` with a dimension index. Dimension 0 is rows, dimension 1 is columns. Summing along dim 1 collapses columns, giving you one sum per row. The result keeps the original number of rows but reduces to one column.

## Solution

```rust
// mean
a.mean_all()?

// row_sums
a.sum(1)?
```
