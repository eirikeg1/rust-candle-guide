//! Exercise 6: Iris Classification (Using the Utility Library)
//!
//! Classify Iris flowers into 3 species using the embedded dataset.
//! This exercise practices using a shared library, train/test splits,
//! and multi-class evaluation.
//!
//! Run: cargo run --bin ex06_iris

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, loss, Linear, Optimizer, VarBuilder, VarMap};
use rust_machine_learning_testing::datasets::iris;
use rust_machine_learning_testing::helpers;

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    println!("=== Exercise 6: Iris Classification ===\n");

    // -------------------------------------------------------
    // Task 1: Load the Iris dataset
    //
    // Use the utility library to load train/test splits.
    //
    // Hint: The iris module has a function that returns pre-split data. See hints/ex06/task1.md if stuck.
    // -------------------------------------------------------
    let (x_train, y_train, x_test, y_test): (Tensor, Tensor, Tensor, Tensor) =
        todo!("Load Iris dataset with train/test split");

    let n_train = x_train.dims()[0];
    let n_test = x_test.dims()[0];
    println!("Iris dataset loaded:");
    println!("  Training:  {n_train} samples, {} features", x_train.dims()[1]);
    println!("  Test:      {n_test} samples");
    println!("  Classes:   {:?}", iris::CLASS_NAMES);
    println!("  Features:  {:?}\n", iris::FEATURE_NAMES);

    // -------------------------------------------------------
    // Task 2: Define the IrisNet struct
    //
    // Architecture: 4 → 16 → 16 → 3
    //   layer1: 4 inputs (features) → 16 hidden
    //   layer2: 16 hidden → 16 hidden
    //   layer3: 16 hidden → 3 outputs (classes)
    //
    // Note: struct fields are provided for you.
    // -------------------------------------------------------
    struct IrisNet {
        layer1: Linear,
        layer2: Linear,
        layer3: Linear,
    }

    // -------------------------------------------------------
    // Task 3: Implement IrisNet::new()
    //
    // Hint: Same layer construction pattern as previous exercises. See hints/ex06/task3.md if stuck.
    // -------------------------------------------------------
    impl IrisNet {
        fn new(vb: VarBuilder) -> anyhow::Result<Self> {
            todo!("Create IrisNet with 3 linear layers: 4→16→16→3")
        }
    }

    // -------------------------------------------------------
    // Task 4: Implement Module::forward
    //
    // Hint: Same forward pass pattern as Exercise 5, with relu instead of gelu. See hints/ex06/task4.md if stuck.
    // -------------------------------------------------------
    impl Module for IrisNet {
        fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
            todo!("Forward pass: layer1 → relu → layer2 → relu → layer3")
        }
    }

    // --- Setup ---
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = IrisNet::new(vb)?;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), Default::default())?;

    // --- Training loop ---
    let epochs = 500;

    for epoch in 0..epochs {
        // ---------------------------------------------------
        // Task 5: Training step
        //
        // Hint: Same training pattern as Exercise 4 (classification). See hints/ex06/task5.md if stuck.
        // ---------------------------------------------------
        todo!("Forward pass, cross-entropy loss, and optimizer step");

        // Remove this once Task 5 is filled in:
        let loss = Tensor::zeros((), DType::F32, device)?;

        if epoch % 100 == 0 || epoch == epochs - 1 {
            let loss_val: f32 = loss.to_scalar()?;
            let train_acc = helpers::accuracy(&model.forward(&x_train)?, &y_train)?;
            let test_acc = helpers::accuracy(&model.forward(&x_test)?, &y_test)?;
            println!(
                "  epoch {epoch:>4}  loss={loss_val:.4}  train_acc={train_acc:.1}%  test_acc={test_acc:.1}%"
            );
        }
    }

    // --- Final evaluation ---
    let test_logits = model.forward(&x_test)?;
    let test_acc = helpers::accuracy(&test_logits, &y_test)?;

    println!("\n--- Final Test Accuracy: {test_acc:.1}% ---");
    assert!(
        test_acc > 80.0,
        "Test accuracy should be > 80%, got {test_acc}%"
    );

    // --- Per-class accuracy ---
    let preds = test_logits.argmax(1)?.to_vec1::<u32>()?;
    let targets = y_test.to_vec1::<u32>()?;
    let per_class = helpers::per_class_accuracy(&preds, &targets, 3);

    println!("\n--- Per-Class Accuracy ---");
    for (i, name) in iris::CLASS_NAMES.iter().enumerate() {
        println!("  {name:<12} {:.1}%", per_class[i]);
    }

    // --- Confusion matrix ---
    println!("\n--- Confusion Matrix (Test Set) ---");
    helpers::print_confusion_matrix(&preds, &targets, 3, &iris::CLASS_NAMES);

    println!("\n🎉 Exercise 6 passed! Your network classifies Iris flowers.");
    Ok(())
}
