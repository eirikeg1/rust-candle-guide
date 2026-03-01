# Tensors: The Foundation of Machine Learning

## What Is a Tensor?

A **tensor** is a multi-dimensional array of numbers. The word comes from
mathematics, but in ML it simply means "a container for numerical data with a
specific shape."

| Rank | Name     | Example shape | Everyday analogy          |
|------|----------|---------------|---------------------------|
| 0    | Scalar   | `()`          | A single temperature      |
| 1    | Vector   | `(5,)`        | A row of sensor readings  |
| 2    | Matrix   | `(3, 4)`      | A spreadsheet             |
| 3    | 3-tensor | `(2, 3, 4)`   | A stack of spreadsheets   |
| n    | n-tensor | `(d1,…,dn)`   | Generalization to n axes  |

## Why ML Uses Tensors

1. **Batch processing** — instead of feeding one sample at a time, you stack
   hundreds into a single tensor and process them in one operation. A batch of
   32 images (each 28×28 pixels) becomes a tensor of shape `(32, 28, 28)`.

2. **GPU parallelism** — GPUs contain thousands of simple cores designed for
   the same operation on many numbers. Tensors map naturally to this model:
   "add these two million-element arrays" becomes a single GPU dispatch.

3. **Unified abstraction** — whether your data is text (sequences of token
   IDs), images (grids of pixel values), or tabular (rows × columns), it all
   fits into tensors.

## Shape, DType, and Device

Every tensor has three properties:

- **Shape** — the size along each dimension. A `(3, 4)` tensor has 3 rows and
  4 columns, holding 12 elements total.

- **DType** (data type) — the numeric precision of each element: `F32`
  (32-bit float, the ML default), `F64`, `BF16`, etc. Smaller dtypes use less
  memory but sacrifice precision.

- **Device** — where the data lives: CPU RAM or a specific GPU. Operations
  require all inputs on the same device.

## Memory Layout

Tensors are stored as flat, contiguous arrays in **row-major** order. A 2×3
matrix `[[1, 2, 3], [4, 5, 6]]` is stored in memory as `[1, 2, 3, 4, 5, 6]`.
The shape tells the library how to interpret this flat buffer as a
multi-dimensional grid.

This is why **reshape** is essentially free — it reinterprets the same memory
with a different shape, without moving any data.

## Key Takeaway

Tensors are not just "arrays." They are the interface between your data and the
hardware that processes it. Understanding shape, dtype, and device is essential
because every ML bug eventually traces back to a mismatch in one of these three.
