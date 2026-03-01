//! Exercise 4: Classification (Neural Network with candle-nn)
//!
//! Build a small MLP to solve the XOR problem using candle-nn's
//! Linear layers, VarMap, VarBuilder, cross-entropy loss, and AdamW.
//!
//! Run: cargo run --bin ex04_classification

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, loss, ops, Linear, Optimizer, VarBuilder, VarMap};

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    println!("=== Exercise 4: Classification (XOR) ===\n");

    // --- Generate XOR dataset ---
    // 200 points in 4 clusters around (0,0), (0,1), (1,0), (1,1)
    let n_per_class = 50;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let centers = [(0.0f32, 0.0, 0u32), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)];

    for &(cx, cy, label) in &centers {
        for _ in 0..n_per_class {
            // Simple deterministic spread (no external RNG needed)
            xs.push(cx);
            xs.push(cy);
            ys.push(label);
        }
    }

    // Add noise via Candle's RNG
    let n_total = n_per_class * 4;
    let x_base = Tensor::new(xs.as_slice(), device)?.reshape((n_total, 2))?;
    let noise = Tensor::randn(0f32, 0.2, (n_total, 2), device)?;
    let x_train = (&x_base + &noise)?;
    let y_train = Tensor::new(ys.as_slice(), device)?;

    println!("Dataset: {n_total} points, 2 features, 2 classes (XOR pattern)");
    println!("  x shape: {:?}", x_train.shape());
    println!("  y shape: {:?}", y_train.shape());
    println!();

    // -------------------------------------------------------
    // Task 1: Understand the MLP struct
    //
    // The network has two linear layers:
    //   layer1: 2 inputs -> 16 hidden units
    //   layer2: 16 hidden units -> 2 outputs (classes)
    //
    // Note: struct fields are provided since Rust macros can't
    // expand to struct fields. Study how Linear is used here —
    // you'll create these layers in Task 2.
    // -------------------------------------------------------
    struct Mlp {
        layer1: Linear,
        layer2: Linear,
    }

    // -------------------------------------------------------
    // Task 2: Implement Mlp::new()
    //
    // Use candle_nn::linear() to create each layer.
    // Use vb.pp("layer_name") to namespace the parameters.
    //
    // Hint: Each layer needs an input size, output size, and parameter namespace. See hints/ex04/task2.md if stuck.
    // -------------------------------------------------------
    impl Mlp {
        fn new(vb: VarBuilder) -> anyhow::Result<Self> {
            todo!("Create the two linear layers using candle_nn::linear")
        }
    }

    // -------------------------------------------------------
    // Task 3: Implement Module::forward
    //
    // Forward pass: layer1 -> relu -> layer2
    // The output is raw logits (no softmax — cross_entropy handles it).
    //
    // Hint: Pass data through layers with an activation in between. See hints/ex04/task3.md if stuck.
    // -------------------------------------------------------
    impl Module for Mlp {
        fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
            todo!("Forward pass: layer1 -> relu -> layer2")
        }
    }

    // -------------------------------------------------------
    // Task 4: Study the setup (VarMap, VarBuilder, model, optimizer)
    //
    // VarMap holds all trainable parameters. VarBuilder creates them
    // with a namespace. The model's new() populates the VarMap via vb.
    // Finally, AdamW receives all vars for optimization.
    //
    // Note: This is provided so the exercise compiles — study the
    // pattern, you'll use it yourself in Exercise 5.
    // -------------------------------------------------------
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = Mlp::new(vb)?;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), Default::default())?;

    // --- Training loop ---
    let epochs = 300;

    for epoch in 0..epochs {
        // ---------------------------------------------------
        // Task 5: Training step
        //
        // 1. Forward pass: get logits from model
        // 2. Compute cross-entropy loss
        // 3. Use optimizer.backward_step(&loss)
        //
        // Hint: Predict, compute loss, optimize — the standard training pattern. See hints/ex04/task5.md if stuck.
        // ---------------------------------------------------
        todo!("Forward pass, cross-entropy loss, and optimizer step");

        // Remove this once Task 5 is filled in:
        let loss = Tensor::zeros((), DType::F32, device)?;

        if epoch % 50 == 0 || epoch == epochs - 1 {
            let loss_val: f32 = loss.to_scalar()?;
            let logits = model.forward(&x_train)?;
            let preds = logits.argmax(1)?;
            let correct: u32 = preds
                .to_vec1::<u32>()?
                .iter()
                .zip(y_train.to_vec1::<u32>()?)
                .filter(|(p, t)| p == &t)
                .count() as u32;
            let acc = correct as f32 / n_total as f32 * 100.0;
            println!("  epoch {epoch:>4}  loss={loss_val:.4}  accuracy={acc:.1}%");
        }
    }

    // --- Final evaluation ---
    let logits = model.forward(&x_train)?;
    let preds = logits.argmax(1)?;
    let correct: u32 = preds
        .to_vec1::<u32>()?
        .iter()
        .zip(y_train.to_vec1::<u32>()?)
        .filter(|(p, t)| p == &t)
        .count() as u32;
    let acc = correct as f32 / n_total as f32 * 100.0;

    println!("\n--- Final accuracy: {acc:.1}% ---");
    assert!(acc > 85.0, "Accuracy should be > 85%, got {acc}%");

    // --- XOR corner test ---
    println!("\n--- XOR Corner Test ---");
    println!("  {:>6} {:>6} {:>8} {:>12} {:>12}", "x1", "x2", "class", "P(class=0)", "P(class=1)");
    println!("  {}", "-".repeat(50));
    let corners = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let expected_labels = [0u32, 1, 1, 0];
    for (corner, &expected) in corners.iter().zip(expected_labels.iter()) {
        let input = Tensor::new(&corner[..], device)?.reshape((1, 2))?;
        let logits = model.forward(&input)?;
        let probs = ops::softmax(&logits, 1)?;
        let probs_vec = probs.to_vec2::<f32>()?;
        let pred = logits.argmax(1)?.to_vec1::<u32>()?[0];
        let mark = if pred == expected { "✓" } else { "✗" };
        println!(
            "  {:.1}    {:.1}    {pred} {mark}     {:.4}      {:.4}",
            corner[0], corner[1], probs_vec[0][0], probs_vec[0][1]
        );
    }

    // --- Confusion matrix ---
    let pred_vec = preds.to_vec1::<u32>()?;
    let target_vec = y_train.to_vec1::<u32>()?;
    let class_names = ["Class 0", "Class 1"];

    println!("\n--- Confusion Matrix ---");
    rust_machine_learning_testing::helpers::print_confusion_matrix(
        &pred_vec,
        &target_vec,
        2,
        &class_names,
    );

    println!("\n🎉 Exercise 4 passed! Your MLP learned XOR.");
    Ok(())
}
