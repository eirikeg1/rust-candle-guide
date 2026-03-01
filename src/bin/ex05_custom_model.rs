//! Exercise 5: Custom Model (Sine Approximation)
//!
//! Build a deeper network to approximate y = sin(x).
//! This exercise practices composing multiple layers with activations.
//!
//! Run: cargo run --bin ex05_custom_model

use std::f32::consts::PI;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, Activation, Linear, Optimizer, VarBuilder, VarMap};

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    println!("=== Exercise 5: Custom Model (Sine Approximation) ===\n");

    // --- Synthetic data: y = sin(x) + noise ---
    let n_samples = 200;
    let x_data: Vec<f32> = (0..n_samples)
        .map(|i| -PI + i as f32 * 2.0 * PI / n_samples as f32)
        .collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| x.sin()).collect();

    let x_train = Tensor::new(x_data.as_slice(), device)?.reshape((n_samples, 1))?;
    let y_true = Tensor::new(y_data.as_slice(), device)?.reshape((n_samples, 1))?;
    let noise = Tensor::randn(0f32, 0.05, (n_samples, 1), device)?;
    let y_train = (&y_true + &noise)?;

    println!("Data: y = sin(x) + noise, {n_samples} samples, x in [-π, π]");
    println!("  x shape: {:?}", x_train.shape());
    println!("  y shape: {:?}\n", y_train.shape());

    // -------------------------------------------------------
    // Task 1: Understand the SineNet struct
    //
    // Three linear layers + an activation function:
    //   layer1: 1 -> 32
    //   layer2: 32 -> 32
    //   layer3: 32 -> 1
    //   activation: Activation (e.g., Activation::Gelu)
    //
    // Note: struct fields are provided since Rust macros can't
    // expand to struct fields. Study how they compose layers +
    // an activation — you'll create them in Task 2.
    // -------------------------------------------------------
    struct SineNet {
        layer1: Linear,
        layer2: Linear,
        layer3: Linear,
        activation: Activation,
    }

    // -------------------------------------------------------
    // Task 2: Implement SineNet::new()
    //
    // Use vb.pp("name") to namespace each layer's parameters.
    // Store Activation::Gelu as the activation.
    //
    // Hint: Same pattern as Exercise 4, but with three layers and a stored activation. See hints/ex05/task2.md if stuck.
    // -------------------------------------------------------
    impl SineNet {
        fn new(vb: VarBuilder) -> anyhow::Result<Self> {
            todo!("Create SineNet with 3 linear layers and Gelu activation")
        }
    }

    // -------------------------------------------------------
    // Task 3: Implement Module::forward
    //
    // layer1 -> activation -> layer2 -> activation -> layer3
    //
    // Hint: Alternate between layers and activations. See hints/ex05/task3.md if stuck.
    // -------------------------------------------------------
    impl Module for SineNet {
        fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
            todo!("Forward pass through 3 layers with activation")
        }
    }

    // --- Setup ---
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = SineNet::new(vb)?;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), Default::default())?;

    // --- Training loop ---
    let epochs = 2000;
    let mut loss_history: Vec<f32> = Vec::new();

    for epoch in 0..epochs {
        // ---------------------------------------------------
        // Task 4: Training step
        //
        // 1. Forward pass
        // 2. MSE loss: mean((pred - y_train)^2)
        // 3. optimizer.backward_step(&loss)
        //
        // Hint: Same training loop pattern, but use MSE loss for regression. See hints/ex05/task4.md if stuck.
        // ---------------------------------------------------
        todo!("Forward pass, MSE loss, and optimizer step");

        // Remove this once Task 4 is filled in:
        let loss = Tensor::zeros((), DType::F32, device)?;

        let loss_val: f32 = loss.to_scalar()?;
        loss_history.push(loss_val);

        if epoch % 500 == 0 || epoch == epochs - 1 {
            println!("  epoch {epoch:>5}  loss={loss_val:.6}");
        }
    }

    // --- Evaluation table (expanded with boundary values) ---
    println!("\n--- Prediction vs Ground Truth ---");
    println!("  {:>8} {:>10} {:>10} {:>10}", "x", "sin(x)", "predicted", "error");
    println!("  {}", "-".repeat(42));

    let test_xs: Vec<f32> = vec![
        -PI, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, PI,
    ];
    let mut max_abs_error: f32 = 0.0;
    for &xv in &test_xs {
        let input = Tensor::new(&[xv], device)?.reshape((1, 1))?;
        let pred: f32 = model.forward(&input)?.flatten_all()?.to_vec1::<f32>()?[0];
        let truth = xv.sin();
        let err = (pred - truth).abs();
        max_abs_error = max_abs_error.max(err);
        println!("  {xv:>8.4} {truth:>10.4} {pred:>10.4} {err:>10.4}");
    }

    println!("\n  Max absolute error: {max_abs_error:.6}");

    // --- Check final loss ---
    let final_pred = model.forward(&x_train)?;
    let final_loss: f32 = (&final_pred - &y_train)?
        .sqr()?
        .mean_all()?
        .to_scalar()?;

    println!("  Final MSE loss:    {final_loss:.6}");
    assert!(
        final_loss < 0.05,
        "Final loss should be < 0.05, got {final_loss}"
    );

    // --- Loss summary ---
    let initial_loss = loss_history.first().copied().unwrap_or(0.0);
    let min_loss = loss_history.iter().cloned().fold(f32::INFINITY, f32::min);
    let is_monotonic = loss_history.windows(2).all(|w| w[1] <= w[0] + 1e-6);
    println!("\n--- Loss Summary ---");
    println!("  Initial loss: {initial_loss:.6}");
    println!("  Final loss:   {final_loss:.6}");
    println!("  Min loss:     {min_loss:.6}");
    println!("  Monotonic:    {} (perfectly decreasing)", if is_monotonic { "yes" } else { "no" });

    println!("\n🎉 Exercise 5 passed! Your network learned to approximate sin(x).");
    Ok(())
}
