use eframe::egui;

use crate::app::TrainingState;

pub fn show(ui: &mut egui::Ui, state: &TrainingState) {
    ui.label("Log");
    egui::ScrollArea::vertical()
        .id_salt("log_panel")
        .stick_to_bottom(true)
        .max_height(ui.available_height())
        .show(ui, |ui| {
            for line in &state.log_lines {
                ui.monospace(line);
            }
            if state.log_lines.is_empty() {
                ui.weak("Click Run to start the exercise.");
            }
        });
}
