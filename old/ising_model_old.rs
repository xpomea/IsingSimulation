use rand::prelude::*;
use rand::rngs::SmallRng;

use crate::dynamics::*;

pub enum InitialCondition {
    AllUp,
    AllDown,
    Random,
}

pub struct IsingModel {
    pub l: usize,
    pub lattice: Vec<i32>,
    pub j: f64,
    pub temp: f64,
    pub beta: f64,
    pub energy: i32,
    pub total_spin: i32,
    pub dynamics: Dynamics

    flip_probabilities: [f64; 5],
    rng: SmallRng,
}

impl IsingModel {
    pub fn new(l: usize, temp: f64, j: Option<f64>, ic: InitialCondition) -> Self { 
        let j = j.unwrap_or(1.0);
        // let i_crit_temp = f64::ln(1.0 + f64::sqrt(2.0)) / j / 2.0;

        let mut rng: SmallRng = rand::make_rng();

        let mut lattice = Vec::with_capacity(l * l);
        for _ in 0..l*l {
            let val = match ic {
                InitialCondition::AllUp => 1,
                InitialCondition::AllDown => -1,
                InitialCondition::Random => if rng.random::<bool>() { 1 } else { -1 }
            };
            lattice.push(val);
        }

        let mut model = Self {
            l,
            lattice,
            j,
            temp: 0.0,
            beta: 0.0,
            energy: 0,
            total_spin: 0,
            flip_probabilities: [0.0, 0.0, 0.0, 0.0, 0.0],
            rng,
        };

        model.energy = model.compute_energy_int();
        model.total_spin = model.compute_magnetization();
        model.set_temperature(temp);

        return model;
    }

    pub fn set_temperature(&mut self, temp: f64) {
        self.temp = temp;
        self.beta = 1.0 / temp;
        self.precompute_flip_probabilities();
    }

    pub fn compute_energy_int(&self) -> i32 {
        let mut energy = 0;
        for i in 0..self.l {
            for j in 0..self.l {
                let idx = i * self.l + j;
                let idx_bottom = if i == self.l - 1 { j } else { (i + 1) * self.l + j };
                let idx_right = if j == self.l - 1 { i * self.l } else { i * self.l + j + 1 };

                energy += self.lattice[idx] * (self.lattice[idx_right] + self.lattice[idx_bottom]);
            }
        }
        return energy;
    }

    pub fn compute_magnetization(&self) -> i32 {
        let mut magnetization = 0;
        for i in 0..self.l*self.l {
            magnetization += self.lattice[i];
        }
        return magnetization;
    }

    pub fn magnetization(&self) -> f64 {
        return self.total_spin as f64 / (self.l * self.l) as f64;
    }

    pub fn energy(&self) -> f64 {
        return self.energy as f64 * self.j;
    }

    fn precompute_flip_probabilities(&mut self) {
        self.flip_probabilities = [-8.0, -4.0, 0.0, 4.0, 8.0];
        for i in 0..5 {
            self.flip_probabilities[i] = f64::exp(-self.beta * self.j * self.flip_probabilities[i]);
        }
    }

    fn flip_energy_delta(&self, i: usize, j: usize, idx: usize) -> (i32, f64) {
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
            2 * alignment,
            self.flip_probabilities[(alignment / 2 + 2) as usize],
        );
    }

    pub fn step(&mut self) {
        let idx = self.rng.random_range(0..self.l * self.l);
        let i = idx / self.l;
        let j = idx - i * self.l;

        let (energy_delta, flip_prob) = self.flip_energy_delta(i, j, idx);
        let flipped = self.rng.random_bool(flip_prob.clamp(0.0, 1.0));

        if flipped {
            self.lattice[idx] *= -1;
            self.energy += energy_delta;
            self.total_spin += 2 * self.lattice[idx];
        }
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.l * self.l {
            self.step();
        }
    }
}
