# Tensor Operations: The Building Blocks

## Element-wise Operations

When you add or multiply two tensors of the same shape, the operation is
applied **element by element**. Given tensors A and B of shape `(2, 3)`:

```
A + B → each (i,j): result[i][j] = A[i][j] + B[i][j]
```

This includes addition, subtraction, multiplication, division, and functions
like `exp()`, `sin()`, `sqr()`.

## Broadcasting

When shapes don't match exactly, broadcasting stretches the smaller tensor to
match the larger one (without copying data). Example: adding a `(3,)` vector
to each row of a `(2, 3)` matrix. The rule: dimensions are compatible if they
are equal, or one of them is 1.

## Matrix Multiplication

Matrix multiplication is the backbone of neural networks. For matrices
A `(m, k)` and B `(k, n)`, the result is `(m, n)` where each element is a dot
product of a row from A with a column from B.

A single linear layer `y = Wx + b` is just a matrix multiply followed by an
addition. Every forward pass in a neural network is a chain of matmuls.

**Inner dimensions must match** — if you try to multiply `(2, 3)` × `(4, 2)`,
you'll get a shape error because 3 ≠ 4.

## Reshape

Reshape reinterprets a tensor's memory with a new shape, without moving data.
A `(2, 6)` tensor and a `(3, 4)` tensor have the same 12 elements — reshape
just changes how you index into them.

Constraints: the total number of elements must stay the same.

## Slicing and Indexing

Indexing extracts sub-tensors:
- **Row indexing**: get a single row from a matrix (reduces rank by 1)
- **Column indexing**: get a single column (with range syntax for the other dim)
- **Range slicing**: extract a contiguous sub-block

## Reductions

Reductions **collapse** one or more dimensions by aggregating:

| Operation | What it does                     |
|-----------|----------------------------------|
| `sum`     | Sum elements along an axis       |
| `mean`    | Average elements along an axis   |
| `max`     | Maximum along an axis            |
| `argmax`  | Index of maximum along an axis   |

The `_all` variants (e.g., `mean_all`) collapse every dimension, returning a
scalar. Reductions are essential for loss functions (averaging over a batch)
and predictions (argmax to pick the class with highest score).

## Concatenation

Concatenation joins tensors along an existing dimension. Stacking two `(2, 3)`
tensors along dimension 0 gives `(4, 3)`. All dimensions except the
concatenation axis must match.

## Why This Matters

Every neural network is a sequence of these primitives: matmul for mixing
features, element-wise ops for activations, reductions for loss, and reshape
for connecting layers of different shapes. Understanding these operations is
understanding what neural networks actually *do* to data.
