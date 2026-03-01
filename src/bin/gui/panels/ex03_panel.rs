use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use rust_machine_learning_testing::exercises::ExerciseResult;

use crate::app::TrainingState;

pub fn show(ui: &mut egui::Ui, state: &TrainingState) {
    if !state.running && state.result.is_none() && state.loss_history.is_empty() {
        ui.label("Click Run to train linear regression.");
        return;
    }

    // Get live data or final result data
    let (loss_hist, data_pts, weight, bias, true_w, true_b) = if let Some(ExerciseResult::Ex03(r)) = &state.result {
        (
            r.loss_history.as_slice(),
            r.data_points.as_slice(),
            r.final_weight,
            r.final_bias,
            r.true_weight,
            r.true_bias,
        )
    } else {
        let w = state.weight_history.last().copied().unwrap_or(0.0);
        let b = state.bias_history.last().copied().unwrap_or(0.0);
        (state.loss_history.as_slice(), &[] as &[(f32, f32)], w, b, 3.0f32, 2.0f32)
    };

    ui.columns(2, |cols| {
        // Loss curve
        cols[0].label("Loss over epochs");
        let loss_points: PlotPoints = loss_hist
            .iter()
            .enumerate()
            .filter(|(_, l)| **l > 0.0)
            .map(|(i, &l)| [i as f64, (l as f64).log10()])
            .collect();
        Plot::new("ex03_loss")
            .height(250.0)
            .allow_scroll(false)
            .x_axis_label("Epoch")
            .y_axis_label("Log\u{2081}\u{2080}(Loss)")
            .show(&mut cols[0], |plot_ui| {
                plot_ui.line(Line::new("Loss", loss_points));
            });

        // Scatter + regression line
        cols[1].label("Data & Regression Line");
        Plot::new("ex03_scatter")
            .height(250.0)
            .allow_scroll(false)
            .x_axis_label("x")
            .y_axis_label("y")
            .show(&mut cols[1], |plot_ui| {
                // Data points
                if !data_pts.is_empty() {
                    let scatter: PlotPoints = data_pts
                        .iter()
                        .map(|&(x, y)| [x as f64, y as f64])
                        .collect();
                    plot_ui.points(
                        Points::new("Data", scatter).radius(3.0),
                    );
                }

                // True line
                let true_line: PlotPoints = (0..100)
                    .map(|i| {
                        let x = i as f64 * 5.0 / 100.0;
                        [x, (true_w as f64) * x + true_b as f64]
                    })
                    .collect();
                plot_ui.line(Line::new("True", true_line).width(2.0));

                // Learned line
                let learned_line: PlotPoints = (0..100)
                    .map(|i| {
                        let x = i as f64 * 5.0 / 100.0;
                        [x, (weight as f64) * x + bias as f64]
                    })
                    .collect();
                plot_ui.line(Line::new("Learned", learned_line).width(2.0));
            });
    });

    if state.result.is_some() {
        ui.add_space(4.0);
        ui.label(format!(
            "Final: w={weight:.4} (true={true_w}), b={bias:.4} (true={true_b})"
        ));
    }
}
