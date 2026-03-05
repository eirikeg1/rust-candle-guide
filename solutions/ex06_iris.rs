use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, loss, Linear, Optimizer, VarBuilder, VarMap};

use crate::datasets::iris;
use crate::helpers;

use super::{Ex06Result, ExerciseResult, TrainingMetrics, TrainingUpdate, UpdateSender};

struct IrisNet {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

impl IrisNet {
    fn new(vb: VarBuilder) -> anyhow::Result<Self> {
        let layer1 = linear(4, 16, vb.pp("layer1"))?;
        let layer2 = linear(16, 16, vb.pp("layer2"))?;
        let layer3 = linear(16, 3, vb.pp("layer3"))?;
        Ok(Self {
            layer1,
            layer2,
            layer3,
        })
    }
}

impl Module for IrisNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.layer1.forward(xs)?;
        let h = h.relu()?;
        let h = self.layer2.forward(&h)?;
        let h = h.relu()?;
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

    let (x_train, y_train, x_test, y_test) = iris::load_iris_split(device)?;

    let n_train = x_train.dims()[0];
    let n_test = x_test.dims()[0];
    log(&format!("Iris dataset: {n_train} train, {n_test} test, 4 features, 3 classes"));

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = IrisNet::new(vb)?;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), Default::default())?;

    let epochs = 500;
    let mut loss_history = Vec::new();
    let mut train_accuracy_history = Vec::new();
    let mut test_accuracy_history = Vec::new();

    for epoch in 0..epochs {
        let logits = model.forward(&x_train)?;
        let train_loss = loss::cross_entropy(&logits, &y_train)?;
        optimizer.backward_step(&train_loss)?;

        let loss_val: f32 = train_loss.to_scalar()?;
        let train_acc = helpers::accuracy(&model.forward(&x_train)?, &y_train)?;
        let test_acc = helpers::accuracy(&model.forward(&x_test)?, &y_test)?;

        loss_history.push(loss_val);
        train_accuracy_history.push(train_acc);
        test_accuracy_history.push(test_acc);

        if let Some(tx) = &tx {
            if epoch % 5 == 0 || epoch == epochs - 1 {
                let _ = tx.send(TrainingUpdate::Epoch {
                    epoch,
                    total_epochs: epochs,
                    metrics: TrainingMetrics {
                        loss: loss_val,
                        accuracy: Some(train_acc),
                        test_accuracy: Some(test_acc),
                        ..Default::default()
                    },
                });
            }
        }

        if epoch % 100 == 0 || epoch == epochs - 1 {
            log(&format!(
                "  epoch {epoch:>4}  loss={loss_val:.4}  train_acc={train_acc:.1}%  test_acc={test_acc:.1}%"
            ));
        }
    }

    // Final evaluation
    let test_logits = model.forward(&x_test)?;
    let preds = test_logits.argmax(1)?.to_vec1::<u32>()?;
    let targets = y_test.to_vec1::<u32>()?;

    // Confusion matrix
    let n_classes = 3;
    let mut confusion = vec![vec![0u32; n_classes]; n_classes];
    for (&pred, &actual) in preds.iter().zip(targets.iter()) {
        confusion[actual as usize][pred as usize] += 1;
    }

    let per_class = helpers::per_class_accuracy(&preds, &targets, n_classes);
    let test_acc = helpers::accuracy(&test_logits, &y_test)?;
    log(&format!("\nFinal test accuracy: {test_acc:.1}%"));
    for (i, name) in iris::CLASS_NAMES.iter().enumerate() {
        log(&format!("  {name}: {:.1}%", per_class[i]));
    }

    let result = ExerciseResult::Ex06(Ex06Result {
        loss_history,
        train_accuracy_history,
        test_accuracy_history,
        confusion_matrix: confusion,
        class_names: iris::CLASS_NAMES.iter().map(|s| s.to_string()).collect(),
        per_class_accuracy: per_class,
    });

    if let Some(tx) = &tx {
        let _ = tx.send(TrainingUpdate::Completed(result.clone()));
    }
    Ok(result)
}
