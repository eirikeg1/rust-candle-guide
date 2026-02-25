//! Exercise 1: Tensor Basics
//!
//! Learn how to create tensors — the fundamental data structure in Candle.
//! Fill in each todo!() to complete the exercise.
//!
//! Run: cargo run --bin ex01_tensor_basics

use candle_core::{Device, DType, Tensor};

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    println!("=== Exercise 1: Tensor Basics ===\n");

    // -------------------------------------------------------
    // Task 1: Create a 1D tensor from a Vec<f32>
    //
    // Create a tensor containing [1.0, 2.0, 3.0, 4.0, 5.0]
    // Hint: Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], device)
    // -------------------------------------------------------
    let t1: Tensor = todo!("Create a 1D tensor from [1.0, 2.0, 3.0, 4.0, 5.0]");

    println!("Task 1 — 1D tensor:");
    println!("  tensor: {t1}");
    println!("  shape:  {:?}", t1.shape());
    println!("  dtype:  {:?}", t1.dtype());
    assert_eq!(t1.dims(), &[5]);
    assert_eq!(t1.dtype(), DType::F32);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 2: Create a 2D tensor from a nested array
    //
    // Create a 2×3 tensor:
    //   [[1.0, 2.0, 3.0],
    //    [4.0, 5.0, 6.0]]
    // Hint: Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], device)
    // -------------------------------------------------------
    let t2: Tensor = todo!("Create a 2x3 tensor from nested arrays");

    println!("Task 2 — 2D tensor:");
    println!("  tensor: {t2}");
    println!("  shape:  {:?}", t2.shape());
    assert_eq!(t2.dims(), &[2, 3]);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 3: Create a 3×4 zeros tensor (F32)
    //
    // Hint: Tensor::zeros((3, 4), DType::F32, device)
    // -------------------------------------------------------
    let t3: Tensor = todo!("Create a 3x4 zeros tensor");

    println!("Task 3 — zeros tensor:");
    println!("  tensor: {t3}");
    println!("  shape:  {:?}", t3.shape());
    assert_eq!(t3.dims(), &[3, 4]);
    assert_eq!(t3.dtype(), DType::F32);
    let sum: f32 = t3.sum_all()?.to_scalar()?;
    assert_eq!(sum, 0.0);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 4: Create a 2×2 ones tensor (F64)
    //
    // Hint: Tensor::ones((2, 2), DType::F64, device)
    // -------------------------------------------------------
    let t4: Tensor = todo!("Create a 2x2 ones tensor with F64 dtype");

    println!("Task 4 — ones tensor:");
    println!("  tensor: {t4}");
    println!("  shape:  {:?}", t4.shape());
    assert_eq!(t4.dims(), &[2, 2]);
    assert_eq!(t4.dtype(), DType::F64);
    let sum: f64 = t4.sum_all()?.to_scalar()?;
    assert_eq!(sum, 4.0);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 5: Create a 3×3 random tensor (normal distribution)
    //
    // Hint: Tensor::randn(mean, std, shape, device)
    //       e.g. Tensor::randn(0f32, 1.0, (3, 3), device)
    // -------------------------------------------------------
    let t5: Tensor = todo!("Create a 3x3 random normal tensor (mean=0, std=1)");

    println!("Task 5 — random normal tensor:");
    println!("  tensor: {t5}");
    println!("  shape:  {:?}", t5.shape());
    assert_eq!(t5.dims(), &[3, 3]);
    assert_eq!(t5.dtype(), DType::F32);
    println!("  ✓ passed\n");

    // -------------------------------------------------------
    // Task 6: Create a 2×5 uniform random tensor
    //
    // Hint: Tensor::rand(lo, hi, shape, device)
    //       e.g. Tensor::rand(0f32, 1.0, (2, 5), device)
    // -------------------------------------------------------
    let t6: Tensor = todo!("Create a 2x5 uniform random tensor in [0, 1)");

    println!("Task 6 — random uniform tensor:");
    println!("  tensor: {t6}");
    println!("  shape:  {:?}", t6.shape());
    assert_eq!(t6.dims(), &[2, 5]);
    assert_eq!(t6.dtype(), DType::F32);
    println!("  ✓ passed\n");

    println!("🎉 All tasks in Exercise 1 passed!");
    Ok(())
}
