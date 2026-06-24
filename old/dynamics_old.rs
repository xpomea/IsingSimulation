use crate::IsingModel;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub trait Dynamics {
    fn step(&mut self, model: &mut IsingModel);
    fn sweep(&mut self, model: &mut IsingModel);
}

pub struct MetropolisDynamics {
    flip_probabilities: [f64; 5],
    rng: SmallRng,
}
impl MetropolisDynamics {
    pub fn new() -> Self {
        let rng: SmallRng = rand::make_rng();
        return Self {
            flip_probabilities: [0.0, 0.0, 0.0, 0.0, 0.0],
            rng,
        };
    }

    fn precompute_flip_probabilities(&mut self, model: &IsingModel) {
        self.flip_probabilities = [-8.0, -4.0, 0.0, 4.0, 8.0];
        for i in 0..5 {
            self.flip_probabilities[i] =
                f64::exp(-model.beta * model.j * self.flip_probabilities[i]);
        }
    }

    fn flip_energy_delta(&self, i: usize, j: usize, idx: usize) -> (f64, f64) {
        let idx_top = if i == 0 { (self.l - 1) * self.l + j } else { (i - 1) * self.l + j };
        let idx_bottom = if i == self.l - 1 { j } else { (i + 1) * self.l + j };
        let idx_right = if j == self.l - 1 { i * self.l } else { i * self.l + j + 1 };
        let idx_left = if j == 0 { (i + 1) * self.l - 1 } else { i * self.l + j - 1 };

        let sum_neighbors = self.lattice[idx_left]
            + self.lattice[idx_top]
            + self.lattice[idx_right]
            + self.lattice[idx_bottom];

        let alignment = self.lattice[idx] * sum_neighbors;

        return (
            2. * self.j * alignment as f64,
            self.flip_probabilities[(alignment / 2 + 2) as usize],
        );
    }
}
impl Dynamics for MetropolisDynamics {
    fn step(&mut self, model: &mut IsingModel) {
        let idx = rng.random_range(0..l * l);
        let i = idx / l;
        let j = idx - i * l;

        let (energy_delta, flip_prob) = self.flip_energy_delta(i, j, idx);
        let flipped = rng.random_bool(flip_prob.clamp(0.0, 1.0));

        if flipped {
            self.lattice[idx] *= -1;
            self.energy += energy_delta;
            self.total_spin += 2 * self.lattice[idx];
        }
    }

}
