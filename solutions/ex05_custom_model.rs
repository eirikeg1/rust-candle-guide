use std::f32::consts::PI;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, Activation, Linear, Optimizer, VarBuilder, VarMap};

use super::{Ex05Result, ExerciseResult, TrainingMetrics, TrainingUpdate, UpdateSender};

struct SineNet {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
    activation: Activation,
}

impl SineNet {
    fn new(vb: VarBuilder) -> anyhow::Result<Self> {
        let layer1 = linear(1, 32, vb.pp("layer1"))?;
        let layer2 = linear(32, 32, vb.pp("layer2"))?;
        let layer3 = linear(32, 1, vb.pp("layer3"))?;
        Ok(Self {
            layer1,
            layer2,
            layer3,
            activation: Activation::Gelu,
        })
    }
}

impl Module for SineNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.layer1.forward(xs)?;
        let h = self.activation.forward(&h)?;
        let h = self.layer2.forward(&h)?;
        let h = self.activation.forward(&h)?;
        self.layer3.forward(&h)
    }
}

pub fn run(tx: Option<UpdateSender>) -> anyhow::Result<ExerciseResult> {
    let device = &Device::Cpu;

    let log = |msg: &str| {
        if let Some(tx) = &tx {
            let _ = tx.send(TrainingUpdate::Log(msg.to_string()));
        }
    };

    let n_samples = 200;
    let x_data: Vec<f32> = (0..n_samples)
        .map(|i| -PI + i as f32 * 2.0 * PI / n_samples as f32)
        .collect();
    let y_true_data: Vec<f32> = x_data.iter().map(|&x| x.sin()).collect();

    let x_train = Tensor::new(x_data.as_slice(), device)?.reshape((n_samples, 1))?;
    let y_true = Tensor::new(y_true_data.as_slice(), device)?.reshape((n_samples, 1))?;
    let noise = Tensor::randn(0f32, 0.05, (n_samples, 1), device)?;
    let y_train = (&y_true + &noise)?;

    // Extract data for plotting
    let y_train_vec = y_train.flatten_all()?.to_vec1::<f32>()?;
    let data_points: Vec<(f32, f32)> = x_data.iter().zip(y_train_vec.iter()).map(|(&x, &y)| (x, y)).collect();
    let true_curve: Vec<(f32, f32)> = x_data.iter().zip(y_true_data.iter()).map(|(&x, &y)| (x, y)).collect();

    log(&format!("Data: y = sin(x) + noise, {n_samples} samples, x in [-pi, pi]"));

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = SineNet::new(vb)?;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), Default::default())?;

    let epochs = 2000;
    let mut loss_history = Vec::new();

    for epoch in 0..epochs {
        let pred = model.forward(&x_train)?;
        let loss = (&pred - &y_train)?.sqr()?.mean_all()?;
        optimizer.backward_step(&loss)?;

        let loss_val: f32 = loss.to_scalar()?;
        loss_history.push(loss_val);

        if let Some(tx) = &tx {
            // Send updates less frequently to avoid channel congestion
            if epoch % 10 == 0 || epoch == epochs - 1 {
                let _ = tx.send(TrainingUpdate::Epoch {
                    epoch,
                    total_epochs: epochs,
                    metrics: TrainingMetrics {
                        loss: loss_val,
                        ..Default::default()
                    },
                });
            }
        }

        if epoch % 500 == 0 || epoch == epochs - 1 {
            log(&format!("  epoch {epoch:>5}  loss={loss_val:.6}"));
        }
    }

    // Generate predicted curve
    let eval_xs: Vec<f32> = (0..200)
        .map(|i| -PI + i as f32 * 2.0 * PI / 200.0)
        .collect();
    let eval_input = Tensor::new(eval_xs.as_slice(), device)?.reshape((200, 1))?;
    let eval_preds = model.forward(&eval_input)?.flatten_all()?.to_vec1::<f32>()?;
    let predicted_curve: Vec<(f32, f32)> = eval_xs.iter().zip(eval_preds.iter()).map(|(&x, &y)| (x, y)).collect();

    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    log(&format!("\nFinal MSE loss: {final_loss:.6}"));

    let result = ExerciseResult::Ex05(Ex05Result {
        loss_history,
        true_curve,
        data_points,
        predicted_curve,
    });

    if let Some(tx) = &tx {
        let _ = tx.send(TrainingUpdate::Completed(result.clone()));
    }
    Ok(result)
}
