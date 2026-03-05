//! Exercise 2: Tensor Operations
//!
//! Learn arithmetic, matmul, reshape, slicing, reductions, and concatenation.
//! Implement the todo!() placeholders in src/exercises/ex02_tensor_ops.rs.
//!
//! Run: cargo run --bin ex02_tensor_ops

use rust_machine_learning_testing::exercises::{self, ExerciseResult};

fn main() -> anyhow::Result<()> {
    println!("=== Exercise 2: Tensor Operations ===\n");

    let result = exercises::ex02_tensor_ops::run(None)?;
    let ExerciseResult::Ex02(res) = result else {
        anyhow::bail!("Unexpected result variant");
    };

    for op in &res.ops {
        println!("{}", op.name);
        println!("  input:  {}", op.input_desc);
        println!("  output: {}", op.output);
        println!("  shape:  {}", op.shape);
        println!();
    }

    println!("Exercise 2 passed! All {} operations completed.", res.ops.len());
    Ok(())
}
