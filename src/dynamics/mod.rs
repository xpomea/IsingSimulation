use crate::ising_model::IsingModel;

pub mod metropolis;
pub mod kawasaki;
pub mod creutz_kawasaki;

pub use metropolis::MetropolisDynamics;
pub use kawasaki::KawasakiDynamics;
pub use creutz_kawasaki::CreutzKawasakiDynamics;

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

pub enum Dynamics {
    Metropolis(MetropolisDynamics),
    Kawasaki(KawasakiDynamics),
    CreutzKawasaki(CreutzKawasakiDynamics),
}

impl Dynamics {
    pub fn sweep(&mut self, model: &mut IsingModel) {
        match self {
            Dynamics::Metropolis(d) => d.sweep(model),
            Dynamics::Kawasaki(d) => d.sweep(model),
            Dynamics::CreutzKawasaki(d) => d.sweep(model),
        }
    }
}
