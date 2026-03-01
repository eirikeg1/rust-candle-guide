use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use rust_machine_learning_testing::exercises::ExerciseResult;

use crate::app::TrainingState;
use crate::widgets::confusion_matrix;

pub fn show(ui: &mut egui::Ui, state: &TrainingState) {
    if !state.running && state.result.is_none() && state.loss_history.is_empty() {
        ui.label("Click Run to train Iris classifier.");
        return;
    }

    let loss_hist = &state.loss_history;
    let train_acc = &state.accuracy_history;
    let test_acc = &state.test_accuracy_history;

    ui.columns(2, |cols| {
        // Loss curve
        cols[0].label("Loss over epochs");
        let loss_points: PlotPoints = loss_hist
            .iter()
            .enumerate()
            .filter(|(_, l)| **l > 0.0)
            .map(|(i, &l)| [i as f64, (l as f64).log10()])
            .collect();
        Plot::new("ex06_loss")
            .height(200.0)
            .allow_scroll(false)
            .x_axis_label("Epoch")
            .y_axis_label("Log\u{2081}\u{2080}(Loss)")
            .show(&mut cols[0], |plot_ui| {
                plot_ui.line(Line::new("Loss", loss_points));
            });

        // Accuracy curves
        cols[1].label("Accuracy over epochs");
        let train_points: PlotPoints = train_acc
            .iter()
            .enumerate()
            .map(|(i, &a)| [i as f64, a as f64])
            .collect();
        let test_points: PlotPoints = test_acc
            .iter()
            .enumerate()
            .map(|(i, &a)| [i as f64, a as f64])
            .collect();
        Plot::new("ex06_acc")
            .height(200.0)
            .allow_scroll(false)
            .x_axis_label("Epoch")
            .y_axis_label("Accuracy %")
            .show(&mut cols[1], |plot_ui| {
                plot_ui.line(
                    Line::new("Train", train_points)
                        .color(egui::Color32::BLUE),
                );
                plot_ui.line(
                    Line::new("Test", test_points)
                        .color(egui::Color32::from_rgb(255, 165, 0)),
                );
            });
    });

    // Confusion matrix + per-class accuracy
    if let Some(ExerciseResult::Ex06(result)) = &state.result {
        ui.add_space(8.0);
        ui.columns(2, |cols| {
            cols[0].label("Confusion Matrix (Test Set)");
            cols[0].add_space(4.0);
            confusion_matrix::show(
                &mut cols[0],
                &result.confusion_matrix,
                &result.class_names,
            );

            cols[1].label("Per-Class Accuracy");
            cols[1].add_space(4.0);
            egui::Grid::new("ex06_per_class")
                .striped(true)
                .show(&mut cols[1], |ui| {
                    ui.strong("Class");
                    ui.strong("Accuracy");
                    ui.end_row();

                    for (i, name) in result.class_names.iter().enumerate() {
                        ui.label(name);
                        ui.label(format!("{:.1}%", result.per_class_accuracy[i]));
                        ui.end_row();
                    }
                });
        });

        let final_test_acc = result
            .test_accuracy_history
            .last()
            .copied()
            .unwrap_or(0.0);
        ui.add_space(4.0);
        ui.label(format!("Final test accuracy: {final_test_acc:.1}%"));
    }
}
