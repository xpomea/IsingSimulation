use crate::ising_model::IsingModel;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub struct MetropolisDynamics {
    pub temp: f64,
    pub beta: f64,

    flip_probabilities: [f64; 5],
    rng: SmallRng,
}

impl MetropolisDynamics {
    pub fn new(temp: f64) -> Self {
        let rng: SmallRng = rand::make_rng();
        let mut d = Self {
            temp,
            beta: 1.0 / temp,
            flip_probabilities: [0.0; 5],
            rng,
        };
        d.precompute_flip_probabilities();
        d
    }

    pub fn set_temperature(&mut self, temp: f64) {
        self.temp = temp;
        self.beta = 1.0 / temp;
        self.precompute_flip_probabilities();
    }

    fn precompute_flip_probabilities(&mut self) {
        self.flip_probabilities = [-8.0, -4.0, 0.0, 4.0, 8.0];
        for i in 0..5 {
            self.flip_probabilities[i] =
                f64::exp(-self.beta * self.flip_probabilities[i]);
        }
    }
    
    fn step(&mut self, model: &mut IsingModel) {
        let idx = self.rng.random_range(0..model.l * model.l);

        let energy_delta = model.flip_energy_delta(idx);
        let flip_prob = self.flip_probabilities[(energy_delta / 4 + 2) as usize];
        let flipped = self.rng.random_bool(flip_prob.clamp(0.0, 1.0));

        if flipped {
            model.flip(idx, energy_delta);
        }
    }

    pub fn sweep(&mut self, model: &mut IsingModel) {
        for _ in 0..model.l * model.l {
            self.step(model);
        }
    }
}
