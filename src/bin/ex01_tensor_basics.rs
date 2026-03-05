//! Exercise 1: Tensor Basics
//!
//! Learn how to create tensors — the fundamental data structure in Candle.
//! Implement the todo!() placeholders in src/exercises/ex01_tensor_basics.rs.
//!
//! Run: cargo run --bin ex01_tensor_basics

use rust_machine_learning_testing::exercises::{self, ExerciseResult};

fn main() -> anyhow::Result<()> {
    println!("=== Exercise 1: Tensor Basics ===\n");

    let result = exercises::ex01_tensor_basics::run(None)?;
    let ExerciseResult::Ex01(res) = result else {
        anyhow::bail!("Unexpected result variant");
    };

    for task in &res.tasks {
        println!("{}", task.name);
        println!("  tensor: {}", task.value);
        println!("  shape:  {}", task.shape);
        println!("  dtype:  {}", task.dtype);
        println!();
    }

    println!("Exercise 1 passed! All {} tensors created.", res.tasks.len());
    Ok(())
}
