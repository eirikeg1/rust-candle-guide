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
    // Hint:
    //   let weight = Var::from_tensor(&Tensor::zeros((1, 1), DType::F32, device)?)?;
    //   let bias = Var::from_tensor(&Tensor::zeros((1,), DType::F32, device)?)?;
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
        // Hint:
        //   let pred = x.matmul(weight.as_tensor())?.broadcast_add(bias.as_tensor())?;
        // ---------------------------------------------------
        let pred: Tensor = todo!("Forward pass: x @ weight + bias");

        // ---------------------------------------------------
        // Task 3: Compute MSE loss
        //
        // loss = mean((pred - y)^2)
        //
        // Hint:
        //   let diff = (&pred - &y)?;
        //   let loss = diff.sqr()?.mean_all()?;
        // ---------------------------------------------------
        let loss: Tensor = todo!("MSE loss: mean of squared differences");

        // ---------------------------------------------------
        // Task 4: Backward pass and parameter update
        //
        // 1. Call loss.backward() to get gradients.
        // 2. Get the gradient for weight and bias from the GradStore.
        // 3. Update the Vars using set() with: param - lr * grad
        //
        // Hint:
        //   let grads = loss.backward()?;
        //   let w_grad = grads.get(weight.as_tensor())?;
        //   let b_grad = grads.get(bias.as_tensor())?;
        //   weight.set(&(weight.as_tensor() - (w_grad * learning_rate)?)?)?;
        //   bias.set(&(bias.as_tensor() - (b_grad * learning_rate)?)?)?;
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

    println!("\n🎉 Exercise 3 passed! Your model learned a good approximation.");
    Ok(())
}
