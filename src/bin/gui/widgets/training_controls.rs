use eframe::egui;

use crate::app::TrainingState;

/// Shows Run and Reset buttons. Returns (run_clicked, reset_clicked).
pub fn show(ui: &mut egui::Ui, state: &TrainingState) -> (bool, bool) {
    let mut run_clicked = false;
    let mut reset_clicked = false;

    if ui
        .add_enabled(!state.running && state.result.is_none(), egui::Button::new("Reset"))
        .clicked()
    {
        reset_clicked = true;
    }

    if ui
        .add_enabled(!state.running, egui::Button::new("Run"))
        .clicked()
    {
        run_clicked = true;
    }

    (run_clicked, reset_clicked)
}
