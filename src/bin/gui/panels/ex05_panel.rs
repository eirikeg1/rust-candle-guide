use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use rust_machine_learning_testing::exercises::ExerciseResult;

use crate::app::TrainingState;

pub fn show(ui: &mut egui::Ui, state: &TrainingState) {
    if !state.running && state.result.is_none() && state.loss_history.is_empty() {
        ui.label("Click Run to train sine approximation.");
        return;
    }

    let loss_hist = &state.loss_history;

    ui.columns(2, |cols| {
        // Loss curve
        cols[0].label("Loss over epochs");
        let loss_points: PlotPoints = loss_hist
            .iter()
            .enumerate()
            .filter(|(_, l)| **l > 0.0)
            .map(|(i, &l)| [i as f64, (l as f64).log10()])
            .collect();
        Plot::new("ex05_loss")
            .height(250.0)
            .allow_scroll(false)
            .x_axis_label("Epoch")
            .y_axis_label("Log\u{2081}\u{2080}(Loss)")
            .show(&mut cols[0], |plot_ui| {
                plot_ui.line(Line::new("Loss", loss_points));
            });

        // Sine fit plot
        cols[1].label("Sine Approximation");
        Plot::new("ex05_sine")
            .height(250.0)
            .allow_scroll(false)
            .x_axis_label("x")
            .y_axis_label("y")
            .show(&mut cols[1], |plot_ui| {
                if let Some(ExerciseResult::Ex05(result)) = &state.result {
                    // Noisy data points
                    let data: PlotPoints = result
                        .data_points
                        .iter()
                        .map(|&(x, y)| [x as f64, y as f64])
                        .collect();
                    plot_ui.points(
                        Points::new("Training data", data)
                            .radius(2.0)
                            .color(egui::Color32::GRAY),
                    );

                    // True sin(x) curve
                    let true_line: PlotPoints = result
                        .true_curve
                        .iter()
                        .map(|&(x, y)| [x as f64, y as f64])
                        .collect();
                    plot_ui.line(
                        Line::new("sin(x)", true_line)
                            .width(2.0)
                            .color(egui::Color32::GREEN),
                    );

                    // Predicted curve
                    let pred_line: PlotPoints = result
                        .predicted_curve
                        .iter()
                        .map(|&(x, y)| [x as f64, y as f64])
                        .collect();
                    plot_ui.line(
                        Line::new("Predicted", pred_line)
                            .width(2.0)
                            .color(egui::Color32::from_rgb(255, 165, 0)),
                    );
                }
            });
    });

    if let Some(ExerciseResult::Ex05(result)) = &state.result {
        let final_loss = result.loss_history.last().copied().unwrap_or(0.0);
        ui.add_space(4.0);
        ui.label(format!("Final MSE loss: {final_loss:.6}"));
    }
}
