mod app;
mod panels;
mod sidebar;
mod widgets;

use app::MlGuiApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 500.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Candle ML Learning Exercises",
        options,
        Box::new(|cc| Ok(Box::new(MlGuiApp::new(cc)))),
    )
}
