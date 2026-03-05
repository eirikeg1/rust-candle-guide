//! Exercise 3: Linear Regression (Manual Gradient Descent)
//!
//! Train a simple linear model y = w*x + b on synthetic data.
//! Implement the todo!() placeholders in src/exercises/ex03_linear_regression.rs.
//!
//! Run: cargo run --bin ex03_linear_regression

use rust_machine_learning_testing::exercises::{self, ExerciseResult};

fn main() -> anyhow::Result<()> {
    println!("=== Exercise 3: Linear Regression ===\n");

    let result = exercises::ex03_linear_regression::run(None)?;
    let ExerciseResult::Ex03(res) = result else {
        anyhow::bail!("Unexpected result variant");
    };

    println!("--- Results ---");
    println!("  Learned weight: {:.4}  (true: {})", res.final_weight, res.true_weight);
    println!("  Learned bias:   {:.4}  (true: {})", res.final_bias, res.true_bias);

    assert!(
        (res.final_weight - res.true_weight).abs() < 0.5,
        "Weight too far from true value: {} vs {}",
        res.final_weight, res.true_weight
    );
    assert!(
        (res.final_bias - res.true_bias).abs() < 0.5,
        "Bias too far from true value: {} vs {}",
        res.final_bias, res.true_bias
    );

    // Prediction table
    println!("\n--- Prediction Table ---");
    println!("  {:>6} {:>10} {:>10} {:>10}", "x", "true y", "predicted", "error");
    println!("  {}", "-".repeat(40));
    for xv in [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0] {
        let true_y = res.true_weight * xv + res.true_bias;
        let pred_y = res.final_weight * xv + res.final_bias;
        let err = (pred_y - true_y).abs();
        println!("  {xv:>6.1} {true_y:>10.4} {pred_y:>10.4} {err:>10.4}");
    }

    println!("\nExercise 3 passed!");
    Ok(())
}
