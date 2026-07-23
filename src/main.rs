use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::time::Instant;

mod dynamics;
mod ising_model;
use dynamics::Dynamics;
use ising_model::{BoundaryCondition, InitialCondition, IsingModel};

use crate::dynamics::{
    BondSelection, CreutzKawasakiDynamics, CreutzThermalDynamics, DemonReplacementMode,
    KawasakiDynamics, MetropolisDynamics, ReservoirType,
};

#[derive(Clone, Copy, PartialEq)]
enum DynamicsType {
    Metropolis,
    Kawasaki,
    CreutzKawasaki,
    CreutzThermal,
}

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

    history_spin_sum: Vec<i64>,

    last_sweep_time_ms: f64,
    last_draw_time_ms: f64,

    selected_dynamics_type: DynamicsType,
    ui_l: usize,
    ui_bond_selection: BondSelection,
    ui_reservoir_type: ReservoirType,

    ui_temp: f64,
    ui_kawasaki_beta: f64,
    ui_kawasaki_m_plus: f64,
    ui_creutz_m: f64,
    ui_creutz_starting_energy: i32,
    ui_thermal_beta: f64,
    ui_thermal_m: f64,
    ui_demon_replacement: DemonReplacementMode,
}

impl Default for IsingApp {
    fn default() -> Self {
        let l = 40;
        Self {
            model: IsingModel::new(l, InitialCondition::Random, BoundaryCondition::Shifted),
            dynamics: Dynamics::Kawasaki(KawasakiDynamics::new(
                l,
                1.0,
                0.9995,
                BondSelection::Random,
                ReservoirType::Annealed,
            )),
            steps_per_frame: 10,
            is_running: false,
            texture: None,

            time_step: 0.0,
            history_mag: Vec::new(),
            history_energy: Vec::new(),
            history_susceptibility: Vec::new(),

            history_spin_sum: vec![0; l],

            last_sweep_time_ms: 0.0,
            last_draw_time_ms: 0.0,

            selected_dynamics_type: DynamicsType::Kawasaki,
            ui_l: l,
            ui_bond_selection: BondSelection::Random,
            ui_reservoir_type: ReservoirType::Annealed,
            ui_temp: 2.27,
            ui_kawasaki_beta: 1.0,
            ui_kawasaki_m_plus: 0.9995,
            ui_creutz_m: 0.997,
            ui_creutz_starting_energy: 80,
            ui_thermal_beta: 1.0,
            ui_thermal_m: 0.997,
            ui_demon_replacement: DemonReplacementMode::PerStep,
        }
    }
}

impl IsingApp {
    fn reset_col_mag_history(&mut self) {
        self.history_spin_sum = vec![0; self.model.l];
    }

    fn restart(&mut self) {
        self.model = IsingModel::new(
            self.ui_l,
            InitialCondition::Random,
            BoundaryCondition::Shifted,
        );
        self.dynamics = match self.selected_dynamics_type {
            DynamicsType::Metropolis => Dynamics::Metropolis(MetropolisDynamics::new(self.ui_temp)),
            DynamicsType::Kawasaki => Dynamics::Kawasaki(KawasakiDynamics::new(
                self.ui_l,
                self.ui_kawasaki_beta,
                self.ui_kawasaki_m_plus,
                self.ui_bond_selection,
                self.ui_reservoir_type,
            )),
            DynamicsType::CreutzKawasaki => Dynamics::CreutzKawasaki(CreutzKawasakiDynamics::new(
                self.ui_l,
                self.ui_creutz_m,
                self.ui_creutz_starting_energy,
                self.ui_bond_selection,
                self.ui_reservoir_type,
            )),
            DynamicsType::CreutzThermal => Dynamics::CreutzThermal(CreutzThermalDynamics::new(
                self.ui_l,
                self.ui_thermal_m,
                self.ui_thermal_beta,
                self.ui_demon_replacement,
                self.ui_bond_selection,
                self.ui_reservoir_type,
            )),
        };
        self.time_step = 0.0;
        self.history_mag.clear();
        self.history_energy.clear();
        self.history_susceptibility.clear();
        self.reset_col_mag_history();
        self.texture = None;
        self.is_running = false;
    }
}

impl eframe::App for IsingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.is_running {
            let start = Instant::now();
            for _ in 0..self.steps_per_frame {
                self.dynamics.sweep(&mut self.model);

                if self.history_spin_sum.len() != self.model.l {
                    self.history_spin_sum = vec![0; self.model.l];
                }
                for x in 0..self.model.l {
                    let mut sum: i64 = 0;
                    for y in 0..self.model.l {
                        sum += self.model.lattice[y * self.model.l + x] as i64;
                    }
                    self.history_spin_sum[x] += sum;
                }
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
                Dynamics::Kawasaki(d) => d.beta,
                Dynamics::CreutzThermal(d) => d.beta,
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
                egui::Slider::new(&mut self.steps_per_frame, 1..=1000).text("Speed (sweeps/frame)"),
            );

            ui.separator();
            ui.heading("Dynamics Setup");

            egui::ComboBox::from_label("Dynamics")
                .selected_text(match self.selected_dynamics_type {
                    DynamicsType::Metropolis => "Metropolis",
                    DynamicsType::Kawasaki => "Kawasaki",
                    DynamicsType::CreutzKawasaki => "Creutz-Kawasaki",
                    DynamicsType::CreutzThermal => "Creutz-Thermal",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.selected_dynamics_type,
                        DynamicsType::Metropolis,
                        "Metropolis",
                    );
                    ui.selectable_value(
                        &mut self.selected_dynamics_type,
                        DynamicsType::Kawasaki,
                        "Kawasaki",
                    );
                    ui.selectable_value(
                        &mut self.selected_dynamics_type,
                        DynamicsType::CreutzKawasaki,
                        "Creutz-Kawasaki",
                    );
                    ui.selectable_value(
                        &mut self.selected_dynamics_type,
                        DynamicsType::CreutzThermal,
                        "Creutz-Thermal",
                    );
                });

            if self.selected_dynamics_type != DynamicsType::Metropolis {
                egui::ComboBox::from_label("Bond selection")
                    .selected_text(match self.ui_bond_selection {
                        BondSelection::Random => "Random",
                        BondSelection::Checkerboard => "Checkerboard",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.ui_bond_selection,
                            BondSelection::Random,
                            "Random",
                        );
                        ui.selectable_value(
                            &mut self.ui_bond_selection,
                            BondSelection::Checkerboard,
                            "Checkerboard",
                        );
                    });

                egui::ComboBox::from_label("Reservoir type")
                    .selected_text(match self.ui_reservoir_type {
                        ReservoirType::Annealed => "Annealed",
                        ReservoirType::Quenched => "Quenched",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.ui_reservoir_type,
                            ReservoirType::Annealed,
                            "Annealed",
                        );
                        ui.selectable_value(
                            &mut self.ui_reservoir_type,
                            ReservoirType::Quenched,
                            "Quenched",
                        );
                    });
            }

            match self.selected_dynamics_type {
                DynamicsType::Metropolis => {
                    if ui
                        .add(egui::Slider::new(&mut self.ui_temp, 0.1..=5.0).text("Temperature T"))
                        .changed()
                    {
                        if let Dynamics::Metropolis(metro) = &mut self.dynamics {
                            metro.set_temperature(self.ui_temp);
                        }
                    }
                }
                DynamicsType::Kawasaki => {
                    ui.add(egui::Slider::new(&mut self.ui_kawasaki_beta, 0.01..=5.0).text("β"));
                    ui.add(egui::Slider::new(&mut self.ui_kawasaki_m_plus, 0.0..=1.0).text("m+"));
                }
                DynamicsType::CreutzKawasaki => {
                    ui.add(egui::Slider::new(&mut self.ui_creutz_m, 0.0..=1.0).text("m"));
                    ui.add(
                        egui::Slider::new(&mut self.ui_creutz_starting_energy, 0..=200)
                            .text("Demon energy"),
                    );
                }
                DynamicsType::CreutzThermal => {
                    ui.add(egui::Slider::new(&mut self.ui_thermal_beta, 0.01..=5.0).text("β"));
                    ui.add(egui::Slider::new(&mut self.ui_thermal_m, 0.0..=1.0).text("m"));
                    egui::ComboBox::from_label("Demon replacement")
                        .selected_text(match self.ui_demon_replacement {
                            DemonReplacementMode::PerStep => "Per-Step",
                            DemonReplacementMode::PerSweep => "Per-Sweep",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.ui_demon_replacement,
                                DemonReplacementMode::PerStep,
                                "Per-Step",
                            );
                            ui.selectable_value(
                                &mut self.ui_demon_replacement,
                                DemonReplacementMode::PerSweep,
                                "Per-Sweep",
                            );
                        });
                }
            }

            ui.add(egui::Slider::new(&mut self.ui_l, 10..=500).text("Lattice size L"));

            if ui.button("🔄 Restart Simulation").clicked() {
                self.restart();
            }

            ui.separator();
            ui.label("Initialization:");
            if ui.button("Random").clicked() {
                self.model = IsingModel::new(
                    self.model.l,
                    InitialCondition::Random,
                    BoundaryCondition::Shifted,
                );
                self.reset_col_mag_history();
            }
            if ui.button("All +1").clicked() {
                self.model = IsingModel::new(
                    self.model.l,
                    InitialCondition::AllUp,
                    BoundaryCondition::Shifted,
                );
                self.reset_col_mag_history();
            }
            if ui.button("All -1").clicked() {
                self.model = IsingModel::new(
                    self.model.l,
                    InitialCondition::AllDown,
                    BoundaryCondition::Shifted,
                );
                self.reset_col_mag_history();
            }

            ui.separator();
            ui.label(format!("Magnetization: {:.3}", self.model.magnetization()));
            ui.label(format!(
                "Energy: {:.3}",
                self.model.energy as f64 / (self.model.l * self.model.l) as f64
            ));

            match &self.dynamics {
                Dynamics::CreutzKawasaki(creutz) => {
                    let sum_h: i32 = creutz.demons_h.iter().sum();
                    let sum_v: i32 = creutz.demons_v.iter().sum();
                    let total_demons = creutz.demons_h.len() + creutz.demons_v.len();
                    let avg_demon_energy = if total_demons > 0 {
                        (sum_h + sum_v) as f64 / total_demons as f64
                    } else {
                        0.0
                    };
                    ui.label(format!("Avg Demon Energy: {:.3}", avg_demon_energy));
                }
                Dynamics::CreutzThermal(ct) => {
                    let sum_h: i32 = ct.demons_h.iter().sum();
                    let sum_v: i32 = ct.demons_v.iter().sum();
                    let total_demons = ct.demons_h.len() + ct.demons_v.len();
                    let avg_demon_energy = if total_demons > 0 {
                        (sum_h + sum_v) as f64 / total_demons as f64
                    } else {
                        0.0
                    };
                    ui.label(format!("Avg Demon Energy: {:.3}", avg_demon_energy));
                    ui.label(format!("β = {:.3}, m = {:.6}", ct.beta, ct.m));
                }
                Dynamics::Kawasaki(kd) => {
                    ui.label(format!("β = {:.3}", kd.beta));
                    ui.label(format!("m+ = {:.6}", kd.m_plus));
                }
                Dynamics::Metropolis(m) => {
                    ui.label(format!("T = {:.3}", m.temp));
                }
            }

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

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                let lattice_side = 200.0_f32;
                let plot_width = ui.available_width();
                let plot_height = 80.0;
                let spacing = 10.0;

                let start = Instant::now();
                self.draw_lattice(ui, ctx, lattice_side);
                self.last_draw_time_ms = start.elapsed().as_secs_f64() * 1000.0;

                ui.add_space(spacing);

                let col_mag: Vec<f64> = if self.time_step > 0.0
                    && self.history_spin_sum.len() == self.model.l
                {
                    let total_denom = (self.model.l as f64) * self.time_step;
                    self.history_spin_sum
                        .iter()
                        .map(|&s| s as f64 / total_denom)
                        .collect()
                } else {
                    let mut inst = vec![0.0; self.model.l];
                    for x in 0..self.model.l {
                        let mut sum = 0;
                        for y in 0..self.model.l {
                            sum += self.model.lattice[y * self.model.l + x];
                        }
                        inst[x] = sum as f64 / self.model.l as f64;
                    }
                    inst
                };

                // 1. Full magnetization profile
                let full_points: Vec<[f64; 2]> = col_mag
                    .iter()
                    .enumerate()
                    .map(|(x, &m)| [x as f64, m])
                    .collect();
                let line_full = Line::new(PlotPoints::new(full_points))
                    .color(egui::Color32::from_rgb(250, 150, 50));

                Plot::new("vertical_mag_plot")
                    .height(plot_height)
                    .width(plot_width)
                    .include_y(-1.1)
                    .include_y(1.1)
                    .show(ui, |plot_ui| plot_ui.line(line_full));

                ui.add_space(spacing);

                // 2 & 3. Zoomed magnetization profiles (left and right) side-by-side
                let half_l = self.model.l / 2;
                ui.columns(2, |columns| {
                    columns[0].vertical(|ui| {
                        let left_points: Vec<[f64; 2]> = col_mag[..half_l]
                            .iter()
                            .enumerate()
                            .map(|(x, &m)| [x as f64, m])
                            .collect();
                        let line_left = Line::new(PlotPoints::new(left_points))
                            .color(egui::Color32::from_rgb(250, 150, 50));

                        Plot::new("left_zoom_mag_plot")
                            .height(plot_height)
                            .auto_bounds(egui::Vec2b::new(false, false))
                            .show(ui, |plot_ui| {
                                plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                                    [0.0, -1.0],
                                    [half_l as f64, -0.998],
                                ));
                                plot_ui.line(line_left);
                            });
                    });

                    columns[1].vertical(|ui| {
                        let right_points: Vec<[f64; 2]> = col_mag[half_l..]
                            .iter()
                            .enumerate()
                            .map(|(x, &m)| [(half_l + x) as f64, m])
                            .collect();
                        let line_right = Line::new(PlotPoints::new(right_points))
                            .color(egui::Color32::from_rgb(250, 150, 50));

                        Plot::new("right_zoom_mag_plot")
                            .height(plot_height)
                            .auto_bounds(egui::Vec2b::new(false, false))
                            .show(ui, |plot_ui| {
                                plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                                    [half_l as f64, 0.998],
                                    [self.model.l as f64, 1.0],
                                ));
                                plot_ui.line(line_right);
                            });
                    });
                });

                ui.add_space(spacing);

                // 4. Spin current plot
                let current_h_data: Option<&Vec<i32>> = match &self.dynamics {
                    Dynamics::CreutzKawasaki(c) => Some(&c.current_h),
                    Dynamics::Kawasaki(c) => Some(&c.current_h),
                    Dynamics::CreutzThermal(c) => Some(&c.current_h),
                    _ => None,
                };

                if let Some(current_h) = current_h_data {
                    let current_points: Vec<[f64; 2]> = current_h
                        .iter()
                        .enumerate()
                        .map(|(x, &c)| [x as f64, c as f64 / self.time_step])
                        .collect();
                    let line_current = Line::new(PlotPoints::new(current_points))
                        .color(egui::Color32::from_rgb(50, 200, 250));

                    Plot::new("spin_current_plot")
                        .height(plot_height)
                        .width(plot_width)
                        .show(ui, |plot_ui| plot_ui.line(line_current));
                }
            });
        });
    }
}

impl IsingApp {
    fn draw_lattice(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, side: f32) {
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
            ui.image((texture.id(), egui::vec2(side, side)));
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Ising Model Simulation",
        options,
        Box::new(|_cc| Ok(Box::new(IsingApp::default()))),
    )
}
