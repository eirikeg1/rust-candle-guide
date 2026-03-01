//! Exercise 3: Linear Regression (Manual Gradient Descent)
//!
//! Train a simple linear model y = w*x + b on synthetic data.
//! You'll use Var for trainable parameters and call backward() manually.
//!
//! Run: cargo run --bin ex03_linear_regression

use candle_core::{DType, Device, Tensor, Var};

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    println!("=== Exercise 3: Linear Regression ===\n");

    // --- Synthetic data: y = 3*x + 2 + noise ---
    let n_samples = 50;
    let true_weight = 3.0f32;
    let true_bias = 2.0f32;

    // x values: linearly spaced from 0 to 5
    let x_data: Vec<f32> = (0..n_samples).map(|i| i as f32 * 5.0 / n_samples as f32).collect();
    let noise = Tensor::randn(0f32, 0.3, (n_samples,), device)?;
    let x = Tensor::new(x_data.as_slice(), device)?.reshape((n_samples, 1))?;
    let y = (&x * true_weight as f64)?.broadcast_add(&Tensor::new(true_bias, device)?)?;
    let y = (y + noise.reshape((n_samples, 1))?)?;

    println!("Data generated: y = {true_weight}*x + {true_bias} + noise");
    println!("  x shape: {:?}", x.shape());
    println!("  y shape: {:?}", y.shape());
    println!();

    // Hyperparameters
    let learning_rate = 0.01f64;
    let epochs = 200;

    // -------------------------------------------------------
    // Task 1: Create trainable parameters
    //
    // Create a weight Var initialized to 0.0 (shape [1, 1])
    // and a bias Var initialized to 0.0 (shape [1]).
    //
    // Hint: Var wraps a tensor to track gradients. See hints/ex03/task1.md if stuck.
    // -------------------------------------------------------
    let weight: Var = todo!("Create trainable weight Var, shape (1, 1), init 0.0");
    let bias: Var = todo!("Create trainable bias Var, shape (1,), init 0.0");

    println!("Initial weight: {}", weight.as_tensor());
    println!("Initial bias:   {}\n", bias.as_tensor());

    // --- Training loop ---
    for epoch in 0..epochs {
        // ---------------------------------------------------
        // Task 2: Forward pass
        //
        // Compute prediction = x @ weight + bias
        //
        // Hint: Combine matrix multiplication and addition. See hints/ex03/task2.md if stuck.
        // ---------------------------------------------------
        let pred: Tensor = todo!("Forward pass: x @ weight + bias");

        // ---------------------------------------------------
        // Task 3: Compute MSE loss
        //
        // loss = mean((pred - y)^2)
        //
        // Hint: Subtract, square, average — the standard MSE formula. See hints/ex03/task3.md if stuck.
        // ---------------------------------------------------
        let loss: Tensor = todo!("MSE loss: mean of squared differences");

        // ---------------------------------------------------
        // Task 4: Backward pass and parameter update
        //
        // 1. Call loss.backward() to get gradients.
        // 2. Get the gradient for weight and bias from the GradStore.
        // 3. Update the Vars using set() with: param - lr * grad
        //
        // Hint: Three steps: get gradients, extract them, update parameters. See hints/ex03/task4.md if stuck.
        // ---------------------------------------------------
        todo!("Backward pass: compute gradients and update weight & bias");

        if epoch % 50 == 0 || epoch == epochs - 1 {
            let loss_val: f32 = loss.to_scalar()?;
            let w: f32 = weight.as_tensor().flatten_all()?.to_vec1::<f32>()?[0];
            let b: f32 = bias.as_tensor().flatten_all()?.to_vec1::<f32>()?[0];
            println!("  epoch {epoch:>4}  loss={loss_val:.4}  w={w:.4}  b={b:.4}");
        }
    }

    // --- Verification ---
    let final_w: f32 = weight.as_tensor().flatten_all()?.to_vec1::<f32>()?[0];
    let final_b: f32 = bias.as_tensor().flatten_all()?.to_vec1::<f32>()?[0];

    println!("\n--- Results ---");
    println!("  Learned weight: {final_w:.4}  (true: {true_weight})");
    println!("  Learned bias:   {final_b:.4}  (true: {true_bias})");

    assert!(
        (final_w - true_weight).abs() < 0.5,
        "Weight too far from true value: {final_w} vs {true_weight}"
    );
    assert!(
        (final_b - true_bias).abs() < 0.5,
        "Bias too far from true value: {final_b} vs {true_bias}"
    );

    // --- Prediction table ---
    println!("\n--- Prediction Table (training range) ---");
    println!("  {:>6} {:>10} {:>10} {:>10}", "x", "true y", "predicted", "error");
    println!("  {}", "-".repeat(40));
    for xv in [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0] {
        let true_y = true_weight * xv + true_bias;
        let pred_y = final_w * xv + final_b;
        let err = (pred_y - true_y).abs();
        println!("  {xv:>6.1} {true_y:>10.4} {pred_y:>10.4} {err:>10.4}");
    }

    // --- Extrapolation table ---
    println!("\n--- Extrapolation Table (unseen x values) ---");
    println!("  {:>6} {:>10} {:>10} {:>10}", "x", "true y", "predicted", "error");
    println!("  {}", "-".repeat(40));
    for xv in [-2.0f32, -1.0, 6.0, 8.0, 10.0] {
        let true_y = true_weight * xv + true_bias;
        let pred_y = final_w * xv + final_b;
        let err = (pred_y - true_y).abs();
        println!("  {xv:>6.1} {true_y:>10.4} {pred_y:>10.4} {err:>10.4}");
    }
    println!("  (Linear models extrapolate perfectly outside training range!)");

    println!("\n🎉 Exercise 3 passed! Your model learned a good approximation.");
    Ok(())
}
