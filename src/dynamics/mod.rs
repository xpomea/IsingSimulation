use crate::ising_model::IsingModel;

pub mod common;
pub mod metropolis;
pub mod kawasaki;
pub mod creutz_kawasaki;
pub mod creutz_thermal;

pub use metropolis::MetropolisDynamics;
pub use kawasaki::KawasakiDynamics;
pub use creutz_kawasaki::CreutzKawasakiDynamics;
pub use creutz_thermal::CreutzThermalDynamics;

#[derive(Clone, Copy, PartialEq)]
pub enum BondSelection {
    Checkerboard,
    Random,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ReservoirType {
    Annealed,
    Quenched,
}

#[derive(Clone, Copy, PartialEq)]
pub enum DemonReplacementMode {
    PerStep,
    PerSweep,
}

pub enum Dynamics {
    Metropolis(MetropolisDynamics),
    Kawasaki(KawasakiDynamics),
    CreutzKawasaki(CreutzKawasakiDynamics),
    CreutzThermal(CreutzThermalDynamics),
}

impl Dynamics {
    pub fn sweep(&mut self, model: &mut IsingModel) {
        match self {
            Dynamics::Metropolis(d) => d.sweep(model),
            Dynamics::Kawasaki(d) => d.sweep(model),
            Dynamics::CreutzKawasaki(d) => d.sweep(model),
            Dynamics::CreutzThermal(d) => d.sweep(model),
        }
    }
}
