//! The Iris dataset — 150 samples, 4 features, 3 classes.
//!
//! This is the classic Fisher/Anderson dataset used in almost every ML
//! introductory course. Features are sepal length, sepal width, petal length,
//! and petal width (all in cm). Classes are Setosa (0), Versicolor (1), and
//! Virginica (2).

use candle_core::{Device, Tensor};

pub const FEATURE_NAMES: [&str; 4] = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
];

pub const CLASS_NAMES: [&str; 3] = ["Setosa", "Versicolor", "Virginica"];

/// Raw Iris data: 150 rows of [sepal_length, sepal_width, petal_length, petal_width, class].
#[rustfmt::skip]
const IRIS_DATA: [[f32; 5]; 150] = [
    // Setosa (class 0) — 50 samples
    [5.1,3.5,1.4,0.2,0.0],[4.9,3.0,1.4,0.2,0.0],[4.7,3.2,1.3,0.2,0.0],[4.6,3.1,1.5,0.2,0.0],
    [5.0,3.6,1.4,0.2,0.0],[5.4,3.9,1.7,0.4,0.0],[4.6,3.4,1.4,0.3,0.0],[5.0,3.4,1.5,0.2,0.0],
    [4.4,2.9,1.4,0.2,0.0],[4.9,3.1,1.5,0.1,0.0],[5.4,3.7,1.5,0.2,0.0],[4.8,3.4,1.6,0.2,0.0],
    [4.8,3.0,1.4,0.1,0.0],[4.3,3.0,1.1,0.1,0.0],[5.8,4.0,1.2,0.2,0.0],[5.7,4.4,1.5,0.4,0.0],
    [5.4,3.9,1.3,0.4,0.0],[5.1,3.5,1.4,0.3,0.0],[5.7,3.8,1.7,0.3,0.0],[5.1,3.8,1.5,0.3,0.0],
    [5.4,3.4,1.7,0.2,0.0],[5.1,3.7,1.5,0.4,0.0],[4.6,3.6,1.0,0.2,0.0],[5.1,3.3,1.7,0.5,0.0],
    [4.8,3.4,1.9,0.2,0.0],[5.0,3.0,1.6,0.2,0.0],[5.0,3.4,1.6,0.4,0.0],[5.2,3.5,1.5,0.2,0.0],
    [5.2,3.4,1.4,0.2,0.0],[4.7,3.2,1.6,0.2,0.0],[4.8,3.1,1.6,0.2,0.0],[5.4,3.4,1.5,0.4,0.0],
    [5.2,4.1,1.5,0.1,0.0],[5.5,4.2,1.4,0.2,0.0],[4.9,3.1,1.5,0.2,0.0],[5.0,3.2,1.2,0.2,0.0],
    [5.5,3.5,1.3,0.2,0.0],[4.9,3.6,1.4,0.1,0.0],[4.4,3.0,1.3,0.2,0.0],[5.1,3.4,1.5,0.2,0.0],
    [5.0,3.5,1.3,0.3,0.0],[4.5,2.3,1.3,0.3,0.0],[4.4,3.2,1.3,0.2,0.0],[5.0,3.5,1.6,0.6,0.0],
    [5.1,3.8,1.9,0.4,0.0],[4.8,3.0,1.4,0.3,0.0],[5.1,3.8,1.6,0.2,0.0],[4.6,3.2,1.4,0.2,0.0],
    [5.3,3.7,1.5,0.2,0.0],[5.0,3.3,1.4,0.2,0.0],
    // Versicolor (class 1) — 50 samples
    [7.0,3.2,4.7,1.4,1.0],[6.4,3.2,4.5,1.5,1.0],[6.9,3.1,4.9,1.5,1.0],[5.5,2.3,4.0,1.3,1.0],
    [6.5,2.8,4.6,1.5,1.0],[5.7,2.8,4.5,1.3,1.0],[6.3,3.3,4.7,1.6,1.0],[4.9,2.4,3.3,1.0,1.0],
    [6.6,2.9,4.6,1.3,1.0],[5.2,2.7,3.9,1.4,1.0],[5.0,2.0,3.5,1.0,1.0],[5.9,3.0,4.2,1.5,1.0],
    [6.0,2.2,4.0,1.0,1.0],[6.1,2.9,4.7,1.4,1.0],[5.6,2.9,3.6,1.3,1.0],[6.7,3.1,4.4,1.4,1.0],
    [5.6,3.0,4.5,1.5,1.0],[5.8,2.7,4.1,1.0,1.0],[6.2,2.2,4.5,1.5,1.0],[5.6,2.5,3.9,1.1,1.0],
    [5.9,3.2,4.8,1.8,1.0],[6.1,2.8,4.0,1.3,1.0],[6.3,2.5,4.9,1.5,1.0],[6.1,2.8,4.7,1.2,1.0],
    [6.4,2.9,4.3,1.3,1.0],[6.6,3.0,4.4,1.4,1.0],[6.8,2.8,4.8,1.4,1.0],[6.7,3.0,5.0,1.7,1.0],
    [6.0,2.9,4.5,1.5,1.0],[5.7,2.6,3.5,1.0,1.0],[5.5,2.4,3.8,1.1,1.0],[5.5,2.4,3.7,1.0,1.0],
    [5.8,2.7,3.9,1.2,1.0],[6.0,2.7,5.1,1.6,1.0],[5.4,3.0,4.5,1.5,1.0],[6.0,3.4,4.5,1.6,1.0],
    [6.7,3.1,4.7,1.5,1.0],[6.3,2.3,4.4,1.3,1.0],[5.6,3.0,4.1,1.3,1.0],[5.5,2.5,4.0,1.3,1.0],
    [5.5,2.6,4.4,1.2,1.0],[6.1,3.0,4.6,1.4,1.0],[5.8,2.6,4.0,1.2,1.0],[5.0,2.3,3.3,1.0,1.0],
    [5.6,2.7,4.2,1.3,1.0],[5.7,3.0,4.2,1.2,1.0],[5.7,2.9,4.2,1.3,1.0],[6.2,2.9,4.3,1.3,1.0],
    [5.1,2.5,3.0,1.1,1.0],[5.7,2.8,4.1,1.3,1.0],
    // Virginica (class 2) — 50 samples
    [6.3,3.3,6.0,2.5,2.0],[5.8,2.7,5.1,1.9,2.0],[7.1,3.0,5.9,2.1,2.0],[6.3,2.9,5.6,1.8,2.0],
    [6.5,3.0,5.8,2.2,2.0],[7.6,3.0,6.6,2.1,2.0],[4.9,2.5,4.5,1.7,2.0],[7.3,2.9,6.3,1.8,2.0],
    [6.7,2.5,5.8,1.8,2.0],[7.2,3.6,6.1,2.5,2.0],[6.5,3.2,5.1,2.0,2.0],[6.4,2.7,5.3,1.9,2.0],
    [6.8,3.0,5.5,2.1,2.0],[5.7,2.5,5.0,2.0,2.0],[5.8,2.8,5.1,2.4,2.0],[6.4,3.2,5.3,2.3,2.0],
    [6.5,3.0,5.5,1.8,2.0],[7.7,3.8,6.7,2.2,2.0],[7.7,2.6,6.9,2.3,2.0],[6.0,2.2,5.0,1.5,2.0],
    [6.9,3.2,5.7,2.3,2.0],[5.6,2.8,4.9,2.0,2.0],[7.7,2.8,6.7,2.0,2.0],[6.3,2.7,4.9,1.8,2.0],
    [6.7,3.3,5.7,2.1,2.0],[7.2,3.2,6.0,1.8,2.0],[6.2,2.8,4.8,1.8,2.0],[6.1,3.0,4.9,1.8,2.0],
    [6.4,2.8,5.6,2.1,2.0],[7.2,3.0,5.8,1.6,2.0],[7.4,2.8,6.1,1.9,2.0],[7.9,3.8,6.4,2.0,2.0],
    [6.4,2.8,5.6,2.2,2.0],[6.3,2.8,5.1,1.5,2.0],[6.1,2.6,5.6,1.4,2.0],[7.7,3.0,6.1,2.3,2.0],
    [6.3,3.4,5.6,2.4,2.0],[6.4,3.1,5.5,1.8,2.0],[6.0,3.0,4.8,1.8,2.0],[6.9,3.1,5.4,2.1,2.0],
    [6.7,3.1,5.6,2.4,2.0],[6.9,3.1,5.1,2.3,2.0],[5.8,2.7,5.1,1.9,2.0],[6.8,3.2,5.9,2.3,2.0],
    [6.7,3.3,5.7,2.5,2.0],[6.7,3.0,5.2,2.3,2.0],[6.3,2.5,5.0,1.9,2.0],[6.5,3.0,5.2,2.0,2.0],
    [6.2,3.4,5.4,2.3,2.0],[5.9,3.0,5.1,1.8,2.0],
];

/// Load the full Iris dataset as tensors.
///
/// Returns `(features, labels)` where features has shape `[150, 4]` and
/// labels has shape `[150]` with values in {0, 1, 2}.
pub fn load_iris(device: &Device) -> anyhow::Result<(Tensor, Tensor)> {
    let mut features = Vec::with_capacity(150 * 4);
    let mut labels = Vec::with_capacity(150);

    for row in &IRIS_DATA {
        features.extend_from_slice(&row[..4]);
        labels.push(row[4] as u32);
    }

    let x = Tensor::new(features.as_slice(), device)?.reshape((150, 4))?;
    let y = Tensor::new(labels.as_slice(), device)?;
    Ok((x, y))
}

/// Load the Iris dataset with a deterministic 80/20 train/test split.
///
/// Uses a fixed interleaved pattern: every 5th sample (indices 4, 9, 14, ...)
/// goes to the test set. This gives 120 training and 30 test samples, with
/// 10 samples from each class in the test set.
///
/// Returns `(x_train, y_train, x_test, y_test)`.
pub fn load_iris_split(
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor, Tensor, Tensor)> {
    let mut train_features = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_features = Vec::new();
    let mut test_labels = Vec::new();

    for (i, row) in IRIS_DATA.iter().enumerate() {
        if i % 5 == 4 {
            test_features.extend_from_slice(&row[..4]);
            test_labels.push(row[4] as u32);
        } else {
            train_features.extend_from_slice(&row[..4]);
            train_labels.push(row[4] as u32);
        }
    }

    let n_train = train_labels.len();
    let n_test = test_labels.len();

    let x_train = Tensor::new(train_features.as_slice(), device)?.reshape((n_train, 4))?;
    let y_train = Tensor::new(train_labels.as_slice(), device)?;
    let x_test = Tensor::new(test_features.as_slice(), device)?.reshape((n_test, 4))?;
    let y_test = Tensor::new(test_labels.as_slice(), device)?;

    Ok((x_train, y_train, x_test, y_test))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_iris_shapes() {
        let device = Device::Cpu;
        let (x, y) = load_iris(&device).unwrap();
        assert_eq!(x.dims(), &[150, 4]);
        assert_eq!(y.dims(), &[150]);
    }

    #[test]
    fn test_load_iris_labels() {
        let device = Device::Cpu;
        let (_, y) = load_iris(&device).unwrap();
        let labels = y.to_vec1::<u32>().unwrap();
        // First 50 should be class 0, next 50 class 1, last 50 class 2
        assert!(labels[..50].iter().all(|&l| l == 0));
        assert!(labels[50..100].iter().all(|&l| l == 1));
        assert!(labels[100..].iter().all(|&l| l == 2));
    }

    #[test]
    fn test_load_iris_split_sizes() {
        let device = Device::Cpu;
        let (x_train, y_train, x_test, y_test) = load_iris_split(&device).unwrap();
        assert_eq!(x_train.dims(), &[120, 4]);
        assert_eq!(y_train.dims(), &[120]);
        assert_eq!(x_test.dims(), &[30, 4]);
        assert_eq!(y_test.dims(), &[30]);
    }

    #[test]
    fn test_load_iris_split_class_balance() {
        let device = Device::Cpu;
        let (_, _, _, y_test) = load_iris_split(&device).unwrap();
        let labels = y_test.to_vec1::<u32>().unwrap();
        let count_0 = labels.iter().filter(|&&l| l == 0).count();
        let count_1 = labels.iter().filter(|&&l| l == 1).count();
        let count_2 = labels.iter().filter(|&&l| l == 2).count();
        // 10 per class in test set
        assert_eq!(count_0, 10);
        assert_eq!(count_1, 10);
        assert_eq!(count_2, 10);
    }
}
