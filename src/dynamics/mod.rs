use crate::ising_model::IsingModel;

pub mod metropolis;
pub mod creutz_kawasaki;

pub use metropolis::MetropolisDynamics;
pub use creutz_kawasaki::CreutzKawasakiDynamics;

pub enum Dynamics {
    Metropolis(MetropolisDynamics),
    CreutzKawasaki(CreutzKawasakiDynamics),
}

impl Dynamics {
    pub fn sweep(&mut self, model: &mut IsingModel) {
        match self {
            Dynamics::Metropolis(d) => d.sweep(model),
            Dynamics::CreutzKawasaki(d) => d.sweep(model),
        }
    }
}
