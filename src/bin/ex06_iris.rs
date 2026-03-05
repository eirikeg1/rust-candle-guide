//! Exercise 6: Iris Classification
//!
//! Classify Iris flowers into 3 species using the embedded dataset.
//! Implement the todo!() placeholders in src/exercises/ex06_iris.rs.
//!
//! Run: cargo run --bin ex06_iris

use rust_machine_learning_testing::datasets::iris;
use rust_machine_learning_testing::exercises::{self, ExerciseResult};
use rust_machine_learning_testing::helpers;

fn main() -> anyhow::Result<()> {
    println!("=== Exercise 6: Iris Classification ===\n");

    let result = exercises::ex06_iris::run(None)?;
    let ExerciseResult::Ex06(res) = result else {
        anyhow::bail!("Unexpected result variant");
    };

    // Final accuracy
    let final_test_acc = res.test_accuracy_history.last().copied().unwrap_or(0.0);
    println!("\n--- Final Test Accuracy: {final_test_acc:.1}% ---");
    assert!(
        final_test_acc > 80.0,
        "Test accuracy should be > 80%, got {final_test_acc}%"
    );

    // Per-class accuracy
    println!("\n--- Per-Class Accuracy ---");
    for (i, name) in iris::CLASS_NAMES.iter().enumerate() {
        println!("  {name:<12} {:.1}%", res.per_class_accuracy[i]);
    }

    // Confusion matrix
    let class_names: Vec<&str> = res.class_names.iter().map(|s| s.as_str()).collect();
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

    println!("\n--- Confusion Matrix (Test Set) ---");
    helpers::print_confusion_matrix(&predictions, &targets, 3, &class_names);

    println!("\nExercise 6 passed! Your network classifies Iris flowers.");
    Ok(())
}
