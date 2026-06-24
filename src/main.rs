use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::time::Instant;

mod dynamics;
mod ising_model;
use dynamics::{Dynamics, MetropolisDynamics};
use ising_model::{InitialCondition, IsingModel, BoundaryCondition};

struct IsingApp {
    model: IsingModel,
    dynamics: Dynamics,
    steps_per_frame: usize,
    is_running: bool,
    texture: Option<egui::TextureHandle>,

    time_step: f64,
    history_mag: Vec<[f64; 2]>,
    history_energy: Vec<[f64; 2]>,
    history_susceptibility: Vec<[f64; 2]>,

    last_sweep_time_ms: f64,
    last_draw_time_ms: f64,
}

impl Default for IsingApp {
    fn default() -> Self {
        return Self {
            model: IsingModel::new(100, InitialCondition::Random, BoundaryCondition::Periodic),
            dynamics: Dynamics::Metropolis(MetropolisDynamics::new(2.269)),
            steps_per_frame: 10,
            is_running: false,
            texture: None,

            time_step: 0.0,
            history_mag: Vec::new(),
            history_energy: Vec::new(),
            history_susceptibility: Vec::new(),

            last_sweep_time_ms: 0.0,
            last_draw_time_ms: 0.0,
        };
    }
}

impl eframe::App for IsingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.is_running {
            let start = Instant::now();
            for _ in 0..self.steps_per_frame {
                self.dynamics.sweep(&mut self.model);
            }
            self.time_step += self.steps_per_frame as f64;
            self.last_sweep_time_ms = start.elapsed().as_secs_f64() * 1000.0;

            let current_mag = self.model.magnetization();
            let current_energy = self.model.energy as f64 / (self.model.l * self.model.l) as f64;

            self.history_mag.push([self.time_step, current_mag]);
            self.history_energy.push([self.time_step, current_energy]);

            let window = 300;
            let mut sum_mag = 0.0;
            let mut sum_mag_squared = 0.0;
            
            let start_idx = self.history_mag.len().saturating_sub(window);
            for i in start_idx..self.history_mag.len() {
                sum_mag += self.history_mag[i][1];
                sum_mag_squared += self.history_mag[i][1].powf(2.0);
            }
            
            let window_len = (self.history_mag.len() - start_idx).max(1) as f64;
            let n = self.model.l * self.model.l;
            
            let beta = match &self.dynamics {
                Dynamics::Metropolis(d) => d.beta,
                _ => 1.0,
            };

            let current_susceptibility = beta
                * (n as f64)
                * (sum_mag_squared / window_len - (sum_mag / window_len).powf(2.0));

            self.history_susceptibility
                .push([self.time_step, current_susceptibility]);

            if self.history_mag.len() > 1000 {
                self.history_mag.remove(0);
                self.history_energy.remove(0);
                self.history_susceptibility.remove(0);
            }

            ctx.request_repaint();
        }

        egui::SidePanel::left("control_panel").show(ctx, |ui| {
            ui.heading("Ising Model Control");
            ui.separator();

            if ui
                .button(if self.is_running { "⏸ Pause" } else { "▶ Start" })
                .clicked()
            {
                self.is_running = !self.is_running;
            }

            ui.add(
                egui::Slider::new(&mut self.steps_per_frame, 1..=100).text("Speed (sweeps/frame)"),
            );

            match &mut self.dynamics {
                Dynamics::Metropolis(metro) => {
                    let mut temp = metro.temp;
                    if ui
                        .add(egui::Slider::new(&mut temp, 0.1..=5.0).text("Temperature T"))
                        .changed()
                    {
                        metro.set_temperature(temp);
                    }
                },
                _ => {}
            }

            ui.separator();
            ui.label("Initialization:");
            if ui.button("Random").clicked() {
                self.model = IsingModel::new(
                    self.model.l,
                    InitialCondition::Random,
                    BoundaryCondition::Shifted,
                );
            }
            if ui.button("All +1").clicked() {
                self.model = IsingModel::new(
                    self.model.l,
                    InitialCondition::AllUp,
                    BoundaryCondition::Shifted,
                );
            }
            if ui.button("All -1").clicked() {
                self.model = IsingModel::new(
                    self.model.l,
                    InitialCondition::AllDown,
                    BoundaryCondition::Shifted,
                );
            }

            ui.separator();
            ui.label(format!("Magnetization: {:.3}", self.model.magnetization()));
            ui.label(format!(
                "Energy: {:.3}",
                self.model.energy as f64 / (self.model.l * self.model.l) as f64
            ));

            ui.separator();
            ui.heading("Профилирование");
            ui.label(format!(
                "Simulation logic: {:.2} ms",
                self.last_sweep_time_ms
            ));
            ui.label(format!("Lattice drawing: {:.2} ms", self.last_draw_time_ms));
            let total = self.last_sweep_time_ms + self.last_draw_time_ms;
            ui.label(format!(
                "Total frametime: {:.2} ms ({:.0} FPS)",
                total,
                1000.0 / (total.max(0.001))
            ));
        });

        egui::SidePanel::right("plots_panel")
            .min_width(300.0)
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.heading("Charts");

                ui.label("Magnetism:");
                let line_mag = Line::new(PlotPoints::new(self.history_mag.clone()))
                    .color(egui::Color32::from_rgb(200, 50, 50));

                Plot::new("mag_plot")
                    .view_aspect(2.0) // Соотношение сторон
                    .include_y(-1.1)
                    .include_y(1.1)
                    .show(ui, |plot_ui| plot_ui.line(line_mag));

                ui.add_space(20.0);

                ui.label("Energy:");
                let line_energy = Line::new(PlotPoints::new(self.history_energy.clone()))
                    .color(egui::Color32::from_rgb(50, 50, 200));

                Plot::new("energy_plot")
                    .view_aspect(2.0)
                    .include_y(-2.1)
                    .include_y(0.1)
                    .show(ui, |plot_ui| plot_ui.line(line_energy));

                ui.add_space(20.0);

                ui.label("Susceptibility:");
                let line_susceptibility =
                    Line::new(PlotPoints::new(self.history_susceptibility.clone()))
                        .color(egui::Color32::from_rgb(50, 200, 50));

                Plot::new("susceptibility_plot")
                    .view_aspect(2.0)
                    .include_y(-0.1)
                    .show(ui, |plot_ui| plot_ui.line(line_susceptibility));
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let start = Instant::now();
            self.draw_lattice(ui, ctx);
            self.last_draw_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        });
    }
}

impl IsingApp {
    fn draw_lattice(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let l = self.model.l;
        let mut pixels = vec![egui::Color32::BLACK; l * l];

        for i in 0..l * l {
            if self.model.lattice[i] == 1 {
                pixels[i] = egui::Color32::from_rgb(218, 38, 28);
            } else {
                pixels[i] = egui::Color32::from_rgb(12, 30, 139);
            }
        }

        let image = egui::ColorImage {
            size: [l, l],
            pixels,
        };

        if let Some(texture) = &mut self.texture {
            texture.set(image, egui::TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture("lattice", image, egui::TextureOptions::NEAREST));
        }

        if let Some(texture) = &self.texture {
            let available_size = ui.available_size();
            let side = available_size.x.min(available_size.y);
            ui.image((texture.id(), egui::vec2(side, side)));
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Ising Model Simulation",
        options,
        Box::new(|_cc| Ok(Box::new(IsingApp::default()))),
    )
}
