use candle_core::{Device, Tensor, Var};

use super::{Ex03Result, ExerciseResult, TrainingMetrics, TrainingUpdate, UpdateSender};

pub fn run(tx: Option<UpdateSender>) -> anyhow::Result<ExerciseResult> {
    let device = &Device::Cpu;

    let log = |msg: &str| {
        if let Some(tx) = &tx {
            let _ = tx.send(TrainingUpdate::Log(msg.to_string()));
        }
    };

    // Synthetic data: y = 3*x + 2 + noise
    let n_samples = 50;
    let true_weight = 3.0f32;
    let true_bias = 2.0f32;

    let x_data: Vec<f32> = (0..n_samples)
        .map(|i| i as f32 * 5.0 / n_samples as f32)
        .collect();
    let noise = Tensor::randn(0f32, 0.3, (n_samples,), device)?;
    let x = Tensor::new(x_data.as_slice(), device)?.reshape((n_samples, 1))?;
    let y = (&x * true_weight as f64)?
        .broadcast_add(&Tensor::new(true_bias, device)?)?;
    let y = (y + noise.reshape((n_samples, 1))?)?;

    // Extract data points for plotting
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = y.flatten_all()?.to_vec1::<f32>()?;
    let data_points: Vec<(f32, f32)> = x_vec.iter().zip(y_vec.iter()).map(|(&x, &y)| (x, y)).collect();

    log(&format!("Data: y = {true_weight}*x + {true_bias} + noise, {n_samples} samples"));

    let learning_rate = 0.01f64;
    let epochs = 200;

    // Trainable parameters
    let weight = Var::new(&[[0.0f32]], device)?;
    let bias = Var::new(&[0.0f32], device)?;

    let mut loss_history = Vec::new();
    let mut weight_history = Vec::new();
    let mut bias_history = Vec::new();

    for epoch in 0..epochs {
        // Forward: pred = x @ weight + bias
        let pred = x.matmul(weight.as_tensor())?.broadcast_add(bias.as_tensor())?;

        // MSE loss
        let diff = (&pred - &y)?;
        let loss = diff.sqr()?.mean_all()?;

        // Backward + update
        let grads = loss.backward()?;
        let grad_w = grads.get(weight.as_tensor()).unwrap();
        let grad_b = grads.get(bias.as_tensor()).unwrap();

        weight.set(&(weight.as_tensor() - (grad_w * learning_rate)?)?)?;
        bias.set(&(bias.as_tensor() - (grad_b * learning_rate)?)?)?;

        let loss_val: f32 = loss.to_scalar()?;
        let w: f32 = weight.as_tensor().flatten_all()?.to_vec1::<f32>()?[0];
        let b: f32 = bias.as_tensor().flatten_all()?.to_vec1::<f32>()?[0];

        loss_history.push(loss_val);
        weight_history.push(w);
        bias_history.push(b);

        if let Some(tx) = &tx {
            let _ = tx.send(TrainingUpdate::Epoch {
                epoch,
                total_epochs: epochs,
                metrics: TrainingMetrics {
                    loss: loss_val,
                    weight: Some(w),
                    bias: Some(b),
                    ..Default::default()
                },
            });
        }

        if epoch % 50 == 0 || epoch == epochs - 1 {
            log(&format!("  epoch {epoch:>4}  loss={loss_val:.4}  w={w:.4}  b={b:.4}"));
        }
    }

    let final_w: f32 = weight.as_tensor().flatten_all()?.to_vec1::<f32>()?[0];
    let final_b: f32 = bias.as_tensor().flatten_all()?.to_vec1::<f32>()?[0];

    log(&format!("\nLearned weight: {final_w:.4}  (true: {true_weight})"));
    log(&format!("Learned bias:   {final_b:.4}  (true: {true_bias})"));

    let result = ExerciseResult::Ex03(Ex03Result {
        loss_history,
        weight_history,
        bias_history,
        data_points,
        final_weight: final_w,
        final_bias: final_b,
        true_weight,
        true_bias,
    });

    if let Some(tx) = &tx {
        let _ = tx.send(TrainingUpdate::Completed(result.clone()));
    }
    Ok(result)
}
