use crate::ising_model::IsingModel;

pub mod metropolis;
pub mod creutz_kawasaki;
pub mod creutz_kawasaki_random;
pub mod kawasaki;
pub mod kawasaki_reservoir_struct;

pub use metropolis::MetropolisDynamics;
pub use creutz_kawasaki::CreutzKawasakiDynamics;
pub use creutz_kawasaki_random::CreutzKawasakiRandomDynamics;
pub use kawasaki::KawasakiDynamics;
pub use kawasaki_reservoir_struct::KawasakiReservoirStructDynamics;

pub enum Dynamics {
    Metropolis(MetropolisDynamics),
    CreutzKawasaki(CreutzKawasakiDynamics),
    CreutzKawasakiRandom(CreutzKawasakiRandomDynamics),
    Kawasaki(KawasakiDynamics),
    KawasakiReservoirStruct(KawasakiReservoirStructDynamics),
}

impl Dynamics {
    pub fn sweep(&mut self, model: &mut IsingModel) {
        match self {
            Dynamics::Metropolis(d) => d.sweep(model),
            Dynamics::CreutzKawasaki(d) => d.sweep(model),
            Dynamics::CreutzKawasakiRandom(d) => d.sweep(model),
            Dynamics::Kawasaki(d) => d.sweep(model),
            Dynamics::KawasakiReservoirStruct(d) => d.sweep(model),
        }
    }
}
