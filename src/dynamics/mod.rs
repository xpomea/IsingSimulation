use crate::ising_model::IsingModel;

pub mod metropolis;

pub use metropolis::MetropolisDynamics;

pub enum Dynamics {
    Metropolis(MetropolisDynamics),
}

impl Dynamics {
    pub fn sweep(&mut self, model: &mut IsingModel) {
        match self {
            Dynamics::Metropolis(d) => d.sweep(model),
        }
    }
}
