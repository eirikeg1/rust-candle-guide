use candle_core::{DType, Device, Tensor};

use super::{Ex01Result, ExerciseResult, TensorInfo, TrainingUpdate, UpdateSender};

pub fn run(tx: Option<UpdateSender>) -> anyhow::Result<ExerciseResult> {
    let device = &Device::Cpu;
    let mut tasks = Vec::new();

    let log = |msg: &str| {
        if let Some(tx) = &tx {
            let _ = tx.send(TrainingUpdate::Log(msg.to_string()));
        }
    };

    // Task 1: 1D tensor from data
    log("Task 1: Creating 1D tensor from [1.0, 2.0, 3.0, 4.0, 5.0]");
    let t1 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], device)?;
    tasks.push(TensorInfo {
        name: "Task 1: 1D from Vec".into(),
        value: format!("{t1}"),
        shape: format!("{:?}", t1.shape()),
        dtype: format!("{:?}", t1.dtype()),
    });
    log(&format!("  Shape: {:?}, dtype: {:?}", t1.shape(), t1.dtype()));

    // Task 2: 2D tensor
    log("Task 2: Creating 2x3 tensor");
    let t2 = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], device)?;
    tasks.push(TensorInfo {
        name: "Task 2: 2D nested".into(),
        value: format!("{t2}"),
        shape: format!("{:?}", t2.shape()),
        dtype: format!("{:?}", t2.dtype()),
    });
    log(&format!("  Shape: {:?}", t2.shape()));

    // Task 3: Zeros
    log("Task 3: Creating 3x4 zeros tensor (F32)");
    let t3 = Tensor::zeros((3, 4), DType::F32, device)?;
    tasks.push(TensorInfo {
        name: "Task 3: Zeros 3x4".into(),
        value: format!("{t3}"),
        shape: format!("{:?}", t3.shape()),
        dtype: format!("{:?}", t3.dtype()),
    });
    log(&format!("  Shape: {:?}, dtype: {:?}", t3.shape(), t3.dtype()));

    // Task 4: Ones F64
    log("Task 4: Creating 2x2 ones tensor (F64)");
    let t4 = Tensor::ones((2, 2), DType::F64, device)?;
    tasks.push(TensorInfo {
        name: "Task 4: Ones 2x2 F64".into(),
        value: format!("{t4}"),
        shape: format!("{:?}", t4.shape()),
        dtype: format!("{:?}", t4.dtype()),
    });
    log(&format!("  Shape: {:?}, dtype: {:?}", t4.shape(), t4.dtype()));

    // Task 5: Random normal
    log("Task 5: Creating 3x3 random normal tensor");
    let t5 = Tensor::randn(0f32, 1.0, (3, 3), device)?;
    tasks.push(TensorInfo {
        name: "Task 5: Randn 3x3".into(),
        value: format!("{t5}"),
        shape: format!("{:?}", t5.shape()),
        dtype: format!("{:?}", t5.dtype()),
    });
    log(&format!("  Shape: {:?}", t5.shape()));

    // Task 6: Random uniform
    log("Task 6: Creating 2x5 uniform random tensor");
    let t6 = Tensor::rand(0f32, 1.0, (2, 5), device)?;
    tasks.push(TensorInfo {
        name: "Task 6: Rand 2x5".into(),
        value: format!("{t6}"),
        shape: format!("{:?}", t6.shape()),
        dtype: format!("{:?}", t6.dtype()),
    });
    log(&format!("  Shape: {:?}", t6.shape()));

    log(&format!("\nAll {} tasks completed.", tasks.len()));

    let result = ExerciseResult::Ex01(Ex01Result { tasks });
    if let Some(tx) = &tx {
        let _ = tx.send(TrainingUpdate::Completed(result.clone()));
    }
    Ok(result)
}
