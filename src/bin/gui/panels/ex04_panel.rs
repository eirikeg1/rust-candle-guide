use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use rust_machine_learning_testing::exercises::ExerciseResult;

use crate::app::TrainingState;
use crate::widgets::confusion_matrix;

pub fn show(ui: &mut egui::Ui, state: &TrainingState) {
    if !state.running && state.result.is_none() && state.loss_history.is_empty() {
        ui.label("Click Run to train XOR classifier.");
        return;
    }

    let loss_hist = &state.loss_history;
    let acc_hist = &state.accuracy_history;

    ui.columns(2, |cols| {
        // Loss curve
        cols[0].label("Loss over epochs");
        let loss_points: PlotPoints = loss_hist
            .iter()
            .enumerate()
            .filter(|(_, l)| **l > 0.0)
            .map(|(i, &l)| [i as f64, (l as f64).log10()])
            .collect();
        Plot::new("ex04_loss")
            .height(200.0)
            .allow_scroll(false)
            .x_axis_label("Epoch")
            .y_axis_label("Log\u{2081}\u{2080}(Loss)")
            .show(&mut cols[0], |plot_ui| {
                plot_ui.line(Line::new("Loss", loss_points));
            });

        // Accuracy curve
        cols[1].label("Accuracy over epochs");
        let acc_points: PlotPoints = acc_hist
            .iter()
            .enumerate()
            .map(|(i, &a)| [i as f64, a as f64])
            .collect();
        Plot::new("ex04_acc")
            .height(200.0)
            .allow_scroll(false)
            .x_axis_label("Epoch")
            .y_axis_label("Accuracy %")
            .show(&mut cols[1], |plot_ui| {
                plot_ui.line(Line::new("Accuracy", acc_points));
            });
    });

    // Decision boundary and confusion matrix when done
    if let Some(ExerciseResult::Ex04(result)) = &state.result {
        ui.add_space(8.0);
        ui.columns(2, |cols| {
            // Decision boundary
            cols[0].label("Decision Boundary");
            Plot::new("ex04_boundary")
                .height(250.0)
                .allow_scroll(false)
                .data_aspect(1.0)
                .x_axis_label("x1")
                .y_axis_label("x2")
                .show(&mut cols[0], |plot_ui| {
                    // Grid background
                    let class0_grid: PlotPoints = result
                        .decision_grid
                        .iter()
                        .filter(|p| p.2 == 0)
                        .map(|p| [p.0 as f64, p.1 as f64])
                        .collect();
                    let class1_grid: PlotPoints = result
                        .decision_grid
                        .iter()
                        .filter(|p| p.2 == 1)
                        .map(|p| [p.0 as f64, p.1 as f64])
                        .collect();

                    plot_ui.points(
                        Points::new("Region 0", class0_grid)
                            .radius(4.0)
                            .color(egui::Color32::from_rgba_unmultiplied(100, 100, 255, 60)),
                    );
                    plot_ui.points(
                        Points::new("Region 1", class1_grid)
                            .radius(4.0)
                            .color(egui::Color32::from_rgba_unmultiplied(255, 100, 100, 60)),
                    );

                    // Data points
                    let data_class0: PlotPoints = result
                        .data_points
                        .iter()
                        .filter(|p| p.2 == 0)
                        .map(|p| [p.0 as f64, p.1 as f64])
                        .collect();
                    let data_class1: PlotPoints = result
                        .data_points
                        .iter()
                        .filter(|p| p.2 == 1)
                        .map(|p| [p.0 as f64, p.1 as f64])
                        .collect();

                    plot_ui.points(
                        Points::new("Class 0", data_class0)
                            .radius(3.0)
                            .color(egui::Color32::BLUE),
                    );
                    plot_ui.points(
                        Points::new("Class 1", data_class1)
                            .radius(3.0)
                            .color(egui::Color32::RED),
                    );
                });

            // Confusion matrix
            cols[1].label("Confusion Matrix");
            cols[1].add_space(4.0);
            confusion_matrix::show(&mut cols[1], &result.confusion_matrix, &result.class_names);
        });
    }
}
