//! Exercise 5: Custom Model (Sine Approximation)
//!
//! Build a deeper network to approximate y = sin(x).
//! Implement the todo!() placeholders in src/exercises/ex05_custom_model.rs.
//!
//! Run: cargo run --bin ex05_custom_model

use rust_machine_learning_testing::exercises::{self, ExerciseResult};

fn main() -> anyhow::Result<()> {
    println!("=== Exercise 5: Custom Model (Sine Approximation) ===\n");

    let result = exercises::ex05_custom_model::run(None)?;
    let ExerciseResult::Ex05(res) = result else {
        anyhow::bail!("Unexpected result variant");
    };

    // Loss summary
    let initial_loss = res.loss_history.first().copied().unwrap_or(0.0);
    let final_loss = res.loss_history.last().copied().unwrap_or(0.0);
    let min_loss = res.loss_history.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("\n--- Loss Summary ---");
    println!("  Initial loss: {initial_loss:.6}");
    println!("  Final loss:   {final_loss:.6}");
    println!("  Min loss:     {min_loss:.6}");

    assert!(
        final_loss < 0.05,
        "Final loss should be < 0.05, got {final_loss}"
    );

    // Sample predictions
    println!("\n--- Sample Predictions ---");
    println!("  {:>8} {:>10} {:>10}", "x", "sin(x)", "predicted");
    println!("  {}", "-".repeat(32));
    for &(x, y_pred) in res.predicted_curve.iter().step_by(20) {
        let y_true = x.sin();
        println!("  {x:>8.4} {y_true:>10.4} {y_pred:>10.4}");
    }

    println!("\nExercise 5 passed! Your network learned to approximate sin(x).");
    Ok(())
}
