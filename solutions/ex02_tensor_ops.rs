use candle_core::{Device, Tensor};

use super::{Ex02Result, ExerciseResult, OpResult, TrainingUpdate, UpdateSender};

pub fn run(tx: Option<UpdateSender>) -> anyhow::Result<ExerciseResult> {
    let device = &Device::Cpu;
    let mut ops = Vec::new();

    let log = |msg: &str| {
        if let Some(tx) = &tx {
            let _ = tx.send(TrainingUpdate::Log(msg.to_string()));
        }
    };

    let a = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], device)?;
    let b = Tensor::new(&[[7.0f32, 8.0, 9.0], [10.0, 11.0, 12.0]], device)?;
    let c = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], device)?;

    // Task 1: Element-wise addition
    log("Task 1: Element-wise addition a + b");
    let add = (&a + &b)?;
    ops.push(OpResult {
        name: "Task 1: a + b".into(),
        input_desc: "a(2x3) + b(2x3)".into(),
        output: format!("{add}"),
        shape: format!("{:?}", add.shape()),
    });
    log(&format!("  Result: {add}"));

    // Task 2: Element-wise multiplication
    log("Task 2: Element-wise multiplication a * b");
    let mul = (&a * &b)?;
    ops.push(OpResult {
        name: "Task 2: a * b".into(),
        input_desc: "a(2x3) * b(2x3)".into(),
        output: format!("{mul}"),
        shape: format!("{:?}", mul.shape()),
    });
    log(&format!("  Result: {mul}"));

    // Task 3: Matrix multiplication
    log("Task 3: Matrix multiplication a @ c");
    let matmul = a.matmul(&c)?;
    ops.push(OpResult {
        name: "Task 3: a @ c".into(),
        input_desc: "a(2x3) @ c(3x2)".into(),
        output: format!("{matmul}"),
        shape: format!("{:?}", matmul.shape()),
    });
    log(&format!("  Result: {matmul}"));

    // Task 4: Reshape
    log("Task 4: Reshape a from (2,3) to (3,2)");
    let reshaped = a.reshape((3, 2))?;
    ops.push(OpResult {
        name: "Task 4: Reshape".into(),
        input_desc: "a(2,3) -> (3,2)".into(),
        output: format!("{reshaped}"),
        shape: format!("{:?}", reshaped.shape()),
    });
    log(&format!("  Result: {reshaped}"));

    // Task 5: Row slicing
    log("Task 5: Extract first row of a");
    let row = a.get(0)?;
    ops.push(OpResult {
        name: "Task 5: Row slice".into(),
        input_desc: "a[0]".into(),
        output: format!("{row}"),
        shape: format!("{:?}", row.shape()),
    });
    log(&format!("  Result: {row}"));

    // Task 6: Column slicing
    log("Task 6: Extract column 1 of a");
    let col = a.narrow(1, 1, 1)?.squeeze(1)?;
    ops.push(OpResult {
        name: "Task 6: Column slice".into(),
        input_desc: "a[:, 1]".into(),
        output: format!("{col}"),
        shape: format!("{:?}", col.shape()),
    });
    log(&format!("  Result: {col}"));

    // Task 7: Reductions
    log("Task 7a: Mean of all elements in a");
    let mean = a.mean_all()?;
    let mean_val: f32 = mean.to_scalar()?;
    ops.push(OpResult {
        name: "Task 7a: Mean all".into(),
        input_desc: "mean(a)".into(),
        output: format!("{mean_val}"),
        shape: "scalar".into(),
    });

    log("Task 7b: Row sums of a");
    let row_sums = a.sum(1)?;
    ops.push(OpResult {
        name: "Task 7b: Row sums".into(),
        input_desc: "sum(a, dim=1)".into(),
        output: format!("{row_sums}"),
        shape: format!("{:?}", row_sums.shape()),
    });
    log(&format!("  Mean: {mean_val}, Row sums: {row_sums}"));

    // Task 8: Concatenation
    log("Task 8: Concatenate a and b along dim 0");
    let stacked = Tensor::cat(&[&a, &b], 0)?;
    ops.push(OpResult {
        name: "Task 8: Cat dim 0".into(),
        input_desc: "cat([a, b], 0)".into(),
        output: format!("{stacked}"),
        shape: format!("{:?}", stacked.shape()),
    });
    log(&format!("  Result: {stacked}"));

    log(&format!("\nAll {} operations completed.", ops.len()));

    let result = ExerciseResult::Ex02(Ex02Result { ops });
    if let Some(tx) = &tx {
        let _ = tx.send(TrainingUpdate::Completed(result.clone()));
    }
    Ok(result)
}
