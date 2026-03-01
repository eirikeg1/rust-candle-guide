use std::sync::mpsc;
use std::thread;

use eframe::egui;
use rust_machine_learning_testing::exercises::{
    self, ExerciseResult, TrainingUpdate,
};

use crate::panels;
use crate::sidebar;
use crate::widgets::training_controls;

/// Which exercise is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exercise {
    Ex01,
    Ex02,
    Ex03,
    Ex04,
    Ex05,
    Ex06,
}

impl Exercise {
    pub fn title(self) -> &'static str {
        match self {
            Self::Ex01 => "Exercise 1: Tensor Basics",
            Self::Ex02 => "Exercise 2: Tensor Operations",
            Self::Ex03 => "Exercise 3: Linear Regression",
            Self::Ex04 => "Exercise 4: XOR Classification",
            Self::Ex05 => "Exercise 5: Sine Approximation",
            Self::Ex06 => "Exercise 6: Iris Classification",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Self::Ex01 => "Ex01 Tensors",
            Self::Ex02 => "Ex02 Ops",
            Self::Ex03 => "Ex03 LinReg",
            Self::Ex04 => "Ex04 XOR",
            Self::Ex05 => "Ex05 Sine",
            Self::Ex06 => "Ex06 Iris",
        }
    }

    pub const ALL: [Exercise; 6] = [
        Self::Ex01,
        Self::Ex02,
        Self::Ex03,
        Self::Ex04,
        Self::Ex05,
        Self::Ex06,
    ];
}

/// Live training state accumulated from channel messages.
pub struct TrainingState {
    pub running: bool,
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss_history: Vec<f32>,
    pub accuracy_history: Vec<f32>,
    pub test_accuracy_history: Vec<f32>,
    pub weight_history: Vec<f32>,
    pub bias_history: Vec<f32>,
    pub log_lines: Vec<String>,
    pub result: Option<ExerciseResult>,
}

impl TrainingState {
    fn new() -> Self {
        Self {
            running: false,
            epoch: 0,
            total_epochs: 0,
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
            test_accuracy_history: Vec::new(),
            weight_history: Vec::new(),
            bias_history: Vec::new(),
            log_lines: Vec::new(),
            result: None,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

pub struct MlGuiApp {
    pub selected: Exercise,
    pub state: TrainingState,
    rx: Option<mpsc::Receiver<TrainingUpdate>>,
}

impl MlGuiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            selected: Exercise::Ex01,
            state: TrainingState::new(),
            rx: None,
        }
    }

    pub fn start_exercise(&mut self, ctx: &egui::Context) {
        if self.state.running {
            return;
        }
        self.state.reset();
        self.state.running = true;

        let (sender, receiver) = mpsc::channel();
        self.rx = Some(receiver);

        let exercise = self.selected;
        let ctx = ctx.clone();

        thread::spawn(move || {
            let result = match exercise {
                Exercise::Ex01 => exercises::ex01_tensor_basics::run(Some(sender.clone())),
                Exercise::Ex02 => exercises::ex02_tensor_ops::run(Some(sender.clone())),
                Exercise::Ex03 => exercises::ex03_linear_regression::run(Some(sender.clone())),
                Exercise::Ex04 => exercises::ex04_classification::run(Some(sender.clone())),
                Exercise::Ex05 => exercises::ex05_custom_model::run(Some(sender.clone())),
                Exercise::Ex06 => exercises::ex06_iris::run(Some(sender.clone())),
            };
            if let Err(e) = result {
                let _ = sender.send(TrainingUpdate::Error(e.to_string()));
            }
            ctx.request_repaint();
        });
    }

    fn drain_channel(&mut self) {
        let Some(rx) = &self.rx else { return };
        while let Ok(update) = rx.try_recv() {
            match update {
                TrainingUpdate::Epoch {
                    epoch,
                    total_epochs,
                    metrics,
                } => {
                    self.state.epoch = epoch;
                    self.state.total_epochs = total_epochs;
                    self.state.loss_history.push(metrics.loss);
                    if let Some(acc) = metrics.accuracy {
                        self.state.accuracy_history.push(acc);
                    }
                    if let Some(acc) = metrics.test_accuracy {
                        self.state.test_accuracy_history.push(acc);
                    }
                    if let Some(w) = metrics.weight {
                        self.state.weight_history.push(w);
                    }
                    if let Some(b) = metrics.bias {
                        self.state.bias_history.push(b);
                    }
                }
                TrainingUpdate::Completed(result) => {
                    self.state.result = Some(result);
                    self.state.running = false;
                }
                TrainingUpdate::Error(msg) => {
                    self.state.log_lines.push(format!("ERROR: {msg}"));
                    self.state.running = false;
                }
                TrainingUpdate::Log(msg) => {
                    self.state.log_lines.push(msg);
                }
            }
        }
    }
}

impl eframe::App for MlGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_channel();

        if self.state.running {
            ctx.request_repaint();
        }

        // Sidebar
        egui::SidePanel::left("sidebar")
            .resizable(true)
            .default_width(160.0)
            .show(ctx, |ui| {
                sidebar::show(ui, &mut self.selected, &self.state);
            });

        // Main panel
        egui::CentralPanel::default().show(ctx, |ui| {
            // Title bar with controls
            ui.horizontal(|ui| {
                ui.heading(self.selected.title());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (run_clicked, reset_clicked) = training_controls::show(ui, &self.state);
                    if run_clicked {
                        self.start_exercise(ctx);
                    }
                    if reset_clicked {
                        self.state.reset();
                        self.rx = None;
                    }
                });
            });
            ui.separator();

            // Split: visualization on top, log on bottom
            let available = ui.available_height();
            let vis_height = (available * 0.7).max(200.0);

            // Visualization area
            egui::Frame::NONE.show(ui, |ui| {
                ui.set_max_height(vis_height);
                egui::ScrollArea::vertical()
                    .id_salt("vis_scroll")
                    .show(ui, |ui| {
                        match self.selected {
                            Exercise::Ex01 => panels::ex01_panel::show(ui, &self.state),
                            Exercise::Ex02 => panels::ex02_panel::show(ui, &self.state),
                            Exercise::Ex03 => panels::ex03_panel::show(ui, &self.state),
                            Exercise::Ex04 => panels::ex04_panel::show(ui, &self.state),
                            Exercise::Ex05 => panels::ex05_panel::show(ui, &self.state),
                            Exercise::Ex06 => panels::ex06_panel::show(ui, &self.state),
                        }
                    });
            });

            ui.separator();

            // Log panel
            panels::log_panel::show(ui, &self.state);
        });
    }
}
