use eframe::egui;
use rust_machine_learning_testing::exercises::ExerciseResult;

use crate::app::TrainingState;

pub fn show(ui: &mut egui::Ui, state: &TrainingState) {
    let result = match &state.result {
        Some(ExerciseResult::Ex01(r)) => r,
        _ => {
            if state.running {
                ui.spinner();
                ui.label("Running...");
            } else {
                ui.label("Click Run to execute tensor basics.");
            }
            return;
        }
    };

    ui.label("Tensor Basics \u{2014} Reference Output");
    ui.add_space(4.0);

    egui::Grid::new("ex01_grid")
        .striped(true)
        .min_col_width(60.0)
        .show(ui, |ui| {
            ui.strong("Task");
            ui.strong("Shape");
            ui.strong("Dtype");
            ui.strong("Value");
            ui.end_row();

            for task in &result.tasks {
                ui.label(&task.name);
                ui.monospace(&task.shape);
                ui.monospace(&task.dtype);
                ui.monospace(&task.value);
                ui.end_row();
            }
        });
}
