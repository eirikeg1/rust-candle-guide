use eframe::egui;

use crate::app::{Exercise, TrainingState};

pub fn show(ui: &mut egui::Ui, selected: &mut Exercise, state: &TrainingState) {
    ui.vertical(|ui| {
        ui.heading("Exercises");
        ui.separator();
        ui.add_space(4.0);

        for ex in Exercise::ALL {
            let is_selected = *selected == ex;
            let label = if is_selected {
                format!("> {}", ex.short())
            } else {
                format!("  {}", ex.short())
            };

            let response = ui.selectable_label(is_selected, label);
            if response.clicked() && !state.running {
                *selected = ex;
            }
        }

        ui.add_space(8.0);
        ui.separator();
        if state.running {
            ui.label("Training...");
            if state.total_epochs > 0 {
                let progress = state.epoch as f32 / state.total_epochs as f32;
                ui.add(egui::ProgressBar::new(progress).show_percentage());
            }
        } else if state.result.is_some() {
            ui.label("Done");
        } else {
            ui.label("Ready");
        }
    });
}
