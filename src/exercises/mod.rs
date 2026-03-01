pub mod ex01_tensor_basics;
pub mod ex02_tensor_ops;
pub mod ex03_linear_regression;
pub mod ex04_classification;
pub mod ex05_custom_model;
pub mod ex06_iris;

use std::sync::mpsc;

/// Alias for the sender side of the training update channel.
pub type UpdateSender = mpsc::Sender<TrainingUpdate>;

/// Messages sent from exercise threads to the GUI.
#[derive(Debug, Clone)]
pub enum TrainingUpdate {
    /// Progress update during training.
    Epoch {
        epoch: usize,
        total_epochs: usize,
        metrics: TrainingMetrics,
    },
    /// Exercise finished successfully.
    Completed(ExerciseResult),
    /// An error occurred.
    Error(String),
    /// A log message.
    Log(String),
}

/// Metrics reported per epoch.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub accuracy: Option<f32>,
    pub test_accuracy: Option<f32>,
    pub weight: Option<f32>,
    pub bias: Option<f32>,
}

/// Structured results returned by each exercise.
#[derive(Debug, Clone)]
pub enum ExerciseResult {
    Ex01(Ex01Result),
    Ex02(Ex02Result),
    Ex03(Ex03Result),
    Ex04(Ex04Result),
    Ex05(Ex05Result),
    Ex06(Ex06Result),
}

// --- Per-exercise result types ---

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub value: String,
    pub shape: String,
    pub dtype: String,
}

#[derive(Debug, Clone)]
pub struct Ex01Result {
    pub tasks: Vec<TensorInfo>,
}

#[derive(Debug, Clone)]
pub struct OpResult {
    pub name: String,
    pub input_desc: String,
    pub output: String,
    pub shape: String,
}

#[derive(Debug, Clone)]
pub struct Ex02Result {
    pub ops: Vec<OpResult>,
}

#[derive(Debug, Clone)]
pub struct Ex03Result {
    pub loss_history: Vec<f32>,
    pub weight_history: Vec<f32>,
    pub bias_history: Vec<f32>,
    /// (x, y) data points used for training
    pub data_points: Vec<(f32, f32)>,
    pub final_weight: f32,
    pub final_bias: f32,
    pub true_weight: f32,
    pub true_bias: f32,
}

#[derive(Debug, Clone)]
pub struct Ex04Result {
    pub loss_history: Vec<f32>,
    pub accuracy_history: Vec<f32>,
    /// (x1, x2, label) for scatter plot
    pub data_points: Vec<(f32, f32, u32)>,
    /// (x1, x2, predicted_class) grid for decision boundary
    pub decision_grid: Vec<(f32, f32, u32)>,
    pub grid_resolution: usize,
    /// confusion_matrix[actual][predicted]
    pub confusion_matrix: Vec<Vec<u32>>,
    pub class_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Ex05Result {
    pub loss_history: Vec<f32>,
    /// (x, y_true_sin) ground truth curve
    pub true_curve: Vec<(f32, f32)>,
    /// (x, y_data) noisy training data
    pub data_points: Vec<(f32, f32)>,
    /// (x, y_pred) learned curve
    pub predicted_curve: Vec<(f32, f32)>,
}

#[derive(Debug, Clone)]
pub struct Ex06Result {
    pub loss_history: Vec<f32>,
    pub train_accuracy_history: Vec<f32>,
    pub test_accuracy_history: Vec<f32>,
    pub confusion_matrix: Vec<Vec<u32>>,
    pub class_names: Vec<String>,
    pub per_class_accuracy: Vec<f32>,
}
