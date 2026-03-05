use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, loss, Linear, Optimizer, VarBuilder, VarMap};

use super::{Ex04Result, ExerciseResult, TrainingMetrics, TrainingUpdate, UpdateSender};

struct Mlp {
    layer1: Linear,
    layer2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder) -> anyhow::Result<Self> {
        let layer1 = linear(2, 16, vb.pp("layer1"))?;
        let layer2 = linear(16, 2, vb.pp("layer2"))?;
        Ok(Self { layer1, layer2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.layer1.forward(xs)?;
        let h = h.relu()?;
        self.layer2.forward(&h)
    }
}

pub fn run(tx: Option<UpdateSender>) -> anyhow::Result<ExerciseResult> {
    let device = &Device::Cpu;

    let log = |msg: &str| {
        if let Some(tx) = &tx {
            let _ = tx.send(TrainingUpdate::Log(msg.to_string()));
        }
    };

    // Generate XOR data
    let n_per_class = 50;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let centers = [(0.0f32, 0.0, 0u32), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)];

    for &(cx, cy, label) in &centers {
        for _ in 0..n_per_class {
            xs.push(cx);
            xs.push(cy);
            ys.push(label);
        }
    }

    let n_total = n_per_class * 4;
    let x_base = Tensor::new(xs.as_slice(), device)?.reshape((n_total, 2))?;
    let noise = Tensor::randn(0f32, 0.2, (n_total, 2), device)?;
    let x_train = (&x_base + &noise)?;
    let y_train = Tensor::new(ys.as_slice(), device)?;

    // Extract data points for plotting
    let x_vals = x_train.to_vec2::<f32>()?;
    let y_vals = y_train.to_vec1::<u32>()?;
    let data_points: Vec<(f32, f32, u32)> = x_vals
        .iter()
        .zip(y_vals.iter())
        .map(|(x, &y)| (x[0], x[1], y))
        .collect();

    log(&format!("XOR dataset: {n_total} points, 2 features, 2 classes"));

    // Setup
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = Mlp::new(vb)?;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), Default::default())?;

    let epochs = 300;
    let mut loss_history = Vec::new();
    let mut accuracy_history = Vec::new();

    for epoch in 0..epochs {
        let logits = model.forward(&x_train)?;
        let train_loss = loss::cross_entropy(&logits, &y_train)?;
        optimizer.backward_step(&train_loss)?;

        let loss_val: f32 = train_loss.to_scalar()?;
        let preds = logits.argmax(1)?;
        let correct: u32 = preds
            .to_vec1::<u32>()?
            .iter()
            .zip(y_vals.iter())
            .filter(|(p, t)| p == t)
            .count() as u32;
        let acc = correct as f32 / n_total as f32 * 100.0;

        loss_history.push(loss_val);
        accuracy_history.push(acc);

        if let Some(tx) = &tx {
            let _ = tx.send(TrainingUpdate::Epoch {
                epoch,
                total_epochs: epochs,
                metrics: TrainingMetrics {
                    loss: loss_val,
                    accuracy: Some(acc),
                    ..Default::default()
                },
            });
        }

        if epoch % 50 == 0 || epoch == epochs - 1 {
            log(&format!("  epoch {epoch:>4}  loss={loss_val:.4}  accuracy={acc:.1}%"));
        }
    }

    // Generate decision boundary grid
    let grid_res = 50;
    let mut grid_xs = Vec::new();
    for gy in 0..grid_res {
        for gx in 0..grid_res {
            let x1 = -0.5 + gx as f32 * 2.0 / grid_res as f32;
            let x2 = -0.5 + gy as f32 * 2.0 / grid_res as f32;
            grid_xs.push(x1);
            grid_xs.push(x2);
        }
    }
    let grid_input = Tensor::new(grid_xs.as_slice(), device)?.reshape((grid_res * grid_res, 2))?;
    let grid_logits = model.forward(&grid_input)?;
    let grid_preds = grid_logits.argmax(1)?.to_vec1::<u32>()?;

    let mut decision_grid = Vec::new();
    for gy in 0..grid_res {
        for gx in 0..grid_res {
            let x1 = -0.5 + gx as f32 * 2.0 / grid_res as f32;
            let x2 = -0.5 + gy as f32 * 2.0 / grid_res as f32;
            let idx = gy * grid_res + gx;
            decision_grid.push((x1, x2, grid_preds[idx]));
        }
    }

    // Confusion matrix
    let final_logits = model.forward(&x_train)?;
    let final_preds = final_logits.argmax(1)?.to_vec1::<u32>()?;
    let mut confusion = vec![vec![0u32; 2]; 2];
    for (&pred, &actual) in final_preds.iter().zip(y_vals.iter()) {
        confusion[actual as usize][pred as usize] += 1;
    }

    let acc = final_preds
        .iter()
        .zip(y_vals.iter())
        .filter(|(p, t)| p == t)
        .count() as f32
        / n_total as f32
        * 100.0;
    log(&format!("\nFinal accuracy: {acc:.1}%"));

    let result = ExerciseResult::Ex04(Ex04Result {
        loss_history,
        accuracy_history,
        data_points,
        decision_grid,
        grid_resolution: grid_res,
        confusion_matrix: confusion,
        class_names: vec!["Class 0".into(), "Class 1".into()],
    });

    if let Some(tx) = &tx {
        let _ = tx.send(TrainingUpdate::Completed(result.clone()));
    }
    Ok(result)
}
