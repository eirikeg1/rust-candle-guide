//! Exercise 4: Classification (Neural Network with candle-nn)
//!
//! Build a small MLP to solve the XOR problem.
//! Implement the todo!() placeholders in src/exercises/ex04_classification.rs.
//!
//! Run: cargo run --bin ex04_classification

use rust_machine_learning_testing::exercises::{self, ExerciseResult};
use rust_machine_learning_testing::helpers;

fn main() -> anyhow::Result<()> {
    println!("=== Exercise 4: Classification (XOR) ===\n");

    let result = exercises::ex04_classification::run(None)?;
    let ExerciseResult::Ex04(res) = result else {
        anyhow::bail!("Unexpected result variant");
    };

    // Final accuracy
    let final_acc = res.accuracy_history.last().copied().unwrap_or(0.0);
    println!("\n--- Final accuracy: {final_acc:.1}% ---");
    assert!(final_acc > 85.0, "Accuracy should be > 85%, got {final_acc}%");

    // Confusion matrix
    println!("\n--- Confusion Matrix ---");
    let class_names: Vec<&str> = res.class_names.iter().map(|s| s.as_str()).collect();
    // Flatten confusion matrix into predictions/targets for print helper
    let mut predictions = Vec::new();
    let mut targets = Vec::new();
    for (actual, row) in res.confusion_matrix.iter().enumerate() {
        for (pred, &count) in row.iter().enumerate() {
            for _ in 0..count {
                predictions.push(pred as u32);
                targets.push(actual as u32);
            }
        }
    }
    helpers::print_confusion_matrix(&predictions, &targets, 2, &class_names);

    println!("\nExercise 4 passed! Your MLP learned XOR.");
    Ok(())
}
