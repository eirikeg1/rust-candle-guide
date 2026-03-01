//! Exercise 2: Tensor Operations
//!
//! Learn arithmetic, matmul, reshape, slicing, reductions, and concatenation.
//! Fill in each todo!() to complete the exercise.
//!
//! Run: cargo run --bin ex02_tensor_ops

use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    println!("=== Exercise 2: Tensor Operations ===\n");

    // Pre-built tensors
    // a = [[1, 2, 3],    b = [[7, 8, 9],     c = [[1, 2],
    //      [4, 5, 6]]         [10, 11, 12]]        [3, 4],
    //                                               [5, 6]]
    let a = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], device)?;
    let b = Tensor::new(&[[7.0f32, 8.0, 9.0], [10.0, 11.0, 12.0]], device)?;
    let c = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], device)?;

    println!("a = {a}");
    println!("b = {b}");
    println!("c = {c}\n");

    // -------------------------------------------------------
    // Task 1: Element-wise addition
    //
    // Compute a + b element-wise.
    // Hint: Rust's operator overloading works on tensors. See hints/ex02/task1.md if stuck.
    // -------------------------------------------------------
    let add: Tensor = todo!("Compute element-wise a + b");

    println!("Task 1 — a + b:");
    println!("  {add}");
    let expected = Tensor::new(&[[8.0f32, 10.0, 12.0], [14.0, 16.0, 18.0]], device)?;
    assert_eq!(add.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 2: Element-wise multiplication
    //
    // Compute a * b element-wise.
    // Hint: Same approach as Task 1 with a different operator. See hints/ex02/task2.md if stuck.
    // -------------------------------------------------------
    let mul: Tensor = todo!("Compute element-wise a * b");

    println!("Task 2 — a * b:");
    println!("  {mul}");
    let expected = Tensor::new(&[[7.0f32, 16.0, 27.0], [40.0, 55.0, 72.0]], device)?;
    assert_eq!(mul.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 3: Matrix multiplication
    //
    // Compute a (2×3) @ c (3×2) = result (2×2)
    // Hint: There's a dedicated method for this linear algebra operation. See hints/ex02/task3.md if stuck.
    // -------------------------------------------------------
    let matmul: Tensor = todo!("Compute matrix multiplication a @ c");

    println!("Task 3 — a @ c:");
    println!("  {matmul}");
    assert_eq!(matmul.dims(), &[2, 2]);
    let expected = Tensor::new(&[[22.0f32, 28.0], [49.0, 64.0]], device)?;
    assert_eq!(matmul.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 4: Reshape
    //
    // Reshape a from (2, 3) to (3, 2).
    // Hint: Look for a method that changes shape without changing data. See hints/ex02/task4.md if stuck.
    // -------------------------------------------------------
    let reshaped: Tensor = todo!("Reshape a from (2,3) to (3,2)");

    println!("Task 4 — reshape (2,3) -> (3,2):");
    println!("  {reshaped}");
    assert_eq!(reshaped.dims(), &[3, 2]);
    // Data stays in row-major order: [[1,2],[3,4],[5,6]]
    let expected = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], device)?;
    assert_eq!(reshaped.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 5: Slicing — extract first row
    //
    // Extract row 0 from a → [1, 2, 3]
    // Hint: Candle has an indexing method for extracting sub-tensors. See hints/ex02/task5.md if stuck.
    // -------------------------------------------------------
    let row: Tensor = todo!("Extract the first row of a");

    println!("Task 5 — first row of a:");
    println!("  {row}");
    assert_eq!(row.dims(), &[3]);
    assert_eq!(row.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0]);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 6: Slicing — extract a column
    //
    // Extract column 1 from a → [2, 5]
    // Hint: Same method as Task 5, but you need to select across two dimensions. See hints/ex02/task6.md if stuck.
    // -------------------------------------------------------
    let col: Tensor = todo!("Extract column 1 from a");

    println!("Task 6 — column 1 of a:");
    println!("  {col}");
    assert_eq!(col.dims(), &[2]);
    assert_eq!(col.to_vec1::<f32>()?, vec![2.0, 5.0]);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 7: Reductions
    //
    // (a) Compute the mean of ALL elements in a.
    // (b) Compute the sum along dimension 1 (sum each row).
    //
    // Hint: Look for reduction methods on Tensor. See hints/ex02/task7.md if stuck.
    // -------------------------------------------------------
    let mean: Tensor = todo!("Compute mean of all elements in a");
    let row_sums: Tensor = todo!("Compute sum of each row in a (along dim 1)");

    println!("Task 7 — reductions:");
    println!("  mean_all: {mean}");
    println!("  row_sums: {row_sums}");
    let mean_val: f32 = mean.to_scalar()?;
    assert!((mean_val - 3.5).abs() < 1e-5, "mean should be 3.5, got {mean_val}");
    assert_eq!(row_sums.to_vec1::<f32>()?, vec![6.0, 15.0]);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 8: Concatenation
    //
    // Stack a and b along dimension 0 to get a (4, 3) tensor.
    // Hint: There's a static method for joining tensors. See hints/ex02/task8.md if stuck.
    // -------------------------------------------------------
    let stacked: Tensor = todo!("Concatenate a and b along dim 0");

    println!("Task 8 — cat along dim 0:");
    println!("  {stacked}");
    assert_eq!(stacked.dims(), &[4, 3]);
    let expected = Tensor::new(
        &[
            [1.0f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        device,
    )?;
    assert_eq!(stacked.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    println!("  ✓ passed\n");

    println!("🎉 All tasks in Exercise 2 passed!");
    Ok(())
}
