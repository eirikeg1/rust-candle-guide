//! Shared utilities for evaluating ML models.

use candle_core::Tensor;

/// Compute classification accuracy as a percentage.
///
/// `logits` has shape `[n, num_classes]`, `targets` has shape `[n]`.
pub fn accuracy(logits: &Tensor, targets: &Tensor) -> anyhow::Result<f32> {
    let preds = logits.argmax(1)?;
    let preds = preds.to_vec1::<u32>()?;
    let targets = targets.to_vec1::<u32>()?;
    let correct = preds
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();
    Ok(correct as f32 / preds.len() as f32 * 100.0)
}

/// Print a confusion matrix to stdout.
///
/// `predictions` and `targets` are `Vec<u32>` of class indices.
pub fn print_confusion_matrix(
    predictions: &[u32],
    targets: &[u32],
    n_classes: usize,
    class_names: &[&str],
) {
    let mut matrix = vec![vec![0u32; n_classes]; n_classes];
    for (&pred, &actual) in predictions.iter().zip(targets.iter()) {
        matrix[actual as usize][pred as usize] += 1;
    }

    // Header
    print!("  {:>12}", "Predicted→");
    for name in class_names {
        print!(" {:>10}", name);
    }
    println!();

    print!("  {:>12}", "Actual↓");
    for _ in class_names {
        print!(" {:>10}", "---");
    }
    println!();

    // Rows
    for (i, name) in class_names.iter().enumerate() {
        print!("  {:>12}", name);
        for j in 0..n_classes {
            print!(" {:>10}", matrix[i][j]);
        }
        println!();
    }
}

/// Compute per-class accuracy percentages.
///
/// Returns a Vec where each entry is the accuracy for that class.
pub fn per_class_accuracy(
    predictions: &[u32],
    targets: &[u32],
    n_classes: usize,
) -> Vec<f32> {
    let mut correct = vec![0u32; n_classes];
    let mut total = vec![0u32; n_classes];

    for (&pred, &actual) in predictions.iter().zip(targets.iter()) {
        total[actual as usize] += 1;
        if pred == actual {
            correct[actual as usize] += 1;
        }
    }

    correct
        .iter()
        .zip(total.iter())
        .map(|(&c, &t)| if t > 0 { c as f32 / t as f32 * 100.0 } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        let device = candle_core::Device::Cpu;
        // Logits where argmax matches targets
        let logits = Tensor::new(&[[0.1f32, 0.9], [0.8, 0.2], [0.3, 0.7]], &device).unwrap();
        let targets = Tensor::new(&[1u32, 0, 1], &device).unwrap();
        let acc = accuracy(&logits, &targets).unwrap();
        assert!((acc - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_accuracy_half() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::new(&[[0.1f32, 0.9], [0.1, 0.9]], &device).unwrap();
        let targets = Tensor::new(&[1u32, 0], &device).unwrap();
        let acc = accuracy(&logits, &targets).unwrap();
        assert!((acc - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_confusion_matrix_values() {
        let preds = vec![0u32, 0, 1, 1, 1, 0];
        let targets = vec![0u32, 0, 1, 1, 0, 1];
        // Class 0: 2 correct, 1 misclassified as 1
        // Class 1: 2 correct, 1 misclassified as 0
        let per_class = per_class_accuracy(&preds, &targets, 2);
        assert!((per_class[0] - 66.666664).abs() < 0.01); // 2/3
        assert!((per_class[1] - 66.666664).abs() < 0.01); // 2/3
    }
}
