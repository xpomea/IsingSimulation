use rand::prelude::*;
use rand::rngs::SmallRng;

pub enum InitialCondition {
    AllUp,
    AllDown,
    Random,
}

pub struct IsingModel {
    pub l: usize,
    pub lattice: Vec<i32>,
    pub energy: i32,
    pub total_spin: i32,
    pub neighbors: Vec<[usize; 4]>,
}

impl IsingModel {
    pub fn new(l: usize, ic: InitialCondition) -> Self { 
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
            energy: 0,
            total_spin: 0,
        };

        model.energy = model.compute_energy_int();
        model.total_spin = model.compute_magnetization();

        return model;
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

    pub fn flip_energy_delta(&self, i: usize, j: usize, idx: usize) -> i32 {
        let idx_top = if i == 0 { (self.l - 1) * self.l + j } else { (i - 1) * self.l + j };
        let idx_bottom = if i == self.l - 1 { j } else { (i + 1) * self.l + j };
        let idx_right = if j == self.l - 1 { i * self.l } else { i * self.l + j + 1 };
        let idx_left = if j == 0 { (i + 1) * self.l - 1 } else { i * self.l + j - 1 };

        let sum_neighbors = self.lattice[idx_left]
            + self.lattice[idx_top]
            + self.lattice[idx_right]
            + self.lattice[idx_bottom];

        let alignment = self.lattice[idx] * sum_neighbors;

        return 2 * alignment;
    }

    pub fn flip(&mut self, idx: usize, energy_delta: i32) {
        self.lattice[idx] *= -1;
        self.energy += energy_delta;
        self.total_spin += 2 * self.lattice[idx];
    }

    pub fn swap_energy_delta(&self) -> i32 {

    }

    pub fn swap(&mut self) {
        
    }
}
