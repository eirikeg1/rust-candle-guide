use eframe::egui::{self, Color32, Pos2, Rect, Vec2};

/// Draw a confusion matrix heatmap.
pub fn show(
    ui: &mut egui::Ui,
    matrix: &[Vec<u32>],
    class_names: &[String],
) {
    let n = class_names.len();
    if n == 0 || matrix.len() != n {
        ui.label("No confusion matrix data");
        return;
    }

    let max_val = matrix.iter().flat_map(|row| row.iter()).copied().max().unwrap_or(1).max(1);

    let cell_size = 40.0;
    let label_width = 80.0;
    let header_height = 30.0;

    let total_w = label_width + n as f32 * cell_size;
    let total_h = header_height + n as f32 * cell_size;

    let (response, painter) = ui.allocate_painter(Vec2::new(total_w, total_h), egui::Sense::hover());
    let origin = response.rect.min;

    // Column headers
    for (j, name) in class_names.iter().enumerate() {
        let x = origin.x + label_width + j as f32 * cell_size + cell_size / 2.0;
        let y = origin.y + header_height / 2.0;
        painter.text(
            Pos2::new(x, y),
            egui::Align2::CENTER_CENTER,
            name,
            egui::FontId::proportional(11.0),
            ui.visuals().text_color(),
        );
    }

    // Rows
    for (i, name) in class_names.iter().enumerate() {
        // Row label
        let x = origin.x + label_width - 4.0;
        let y = origin.y + header_height + i as f32 * cell_size + cell_size / 2.0;
        painter.text(
            Pos2::new(x, y),
            egui::Align2::RIGHT_CENTER,
            name,
            egui::FontId::proportional(11.0),
            ui.visuals().text_color(),
        );

        // Cells
        for j in 0..n {
            let val = matrix[i][j];
            let intensity = val as f32 / max_val as f32;

            let color = if i == j {
                // Diagonal: green shading
                Color32::from_rgba_unmultiplied(
                    (40.0 * (1.0 - intensity)) as u8,
                    (100.0 + 155.0 * intensity) as u8,
                    (40.0 * (1.0 - intensity)) as u8,
                    (60.0 + 195.0 * intensity) as u8,
                )
            } else {
                // Off-diagonal: red shading
                Color32::from_rgba_unmultiplied(
                    (100.0 + 155.0 * intensity) as u8,
                    (40.0 * (1.0 - intensity)) as u8,
                    (40.0 * (1.0 - intensity)) as u8,
                    (60.0 + 195.0 * intensity) as u8,
                )
            };

            let cell_origin = Pos2::new(
                origin.x + label_width + j as f32 * cell_size,
                origin.y + header_height + i as f32 * cell_size,
            );
            let cell_rect = Rect::from_min_size(cell_origin, Vec2::splat(cell_size));

            painter.rect_filled(cell_rect, 2.0, color);
            painter.rect_stroke(
                cell_rect,
                2.0,
                ui.visuals().widgets.noninteractive.bg_stroke,
                egui::StrokeKind::Outside,
            );

            painter.text(
                cell_rect.center(),
                egui::Align2::CENTER_CENTER,
                val.to_string(),
                egui::FontId::proportional(13.0),
                Color32::WHITE,
            );
        }
    }
}
