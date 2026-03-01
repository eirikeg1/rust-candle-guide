use eframe::egui;
use rust_machine_learning_testing::exercises::ExerciseResult;

use crate::app::TrainingState;

pub fn show(ui: &mut egui::Ui, state: &TrainingState) {
    let result = match &state.result {
        Some(ExerciseResult::Ex02(r)) => r,
        _ => {
            if state.running {
                ui.spinner();
                ui.label("Running...");
            } else {
                ui.label("Click Run to execute tensor operations.");
            }
            return;
        }
    };

    ui.label("Tensor Operations \u{2014} Reference Output");
    ui.add_space(4.0);

    egui::Grid::new("ex02_grid")
        .striped(true)
        .min_col_width(60.0)
        .show(ui, |ui| {
            ui.strong("Operation");
            ui.strong("Input");
            ui.strong("Shape");
            ui.strong("Output");
            ui.end_row();

            for op in &result.ops {
                ui.label(&op.name);
                ui.monospace(&op.input_desc);
                ui.monospace(&op.shape);
                ui.monospace(&op.output);
                ui.end_row();
            }
        });
}
