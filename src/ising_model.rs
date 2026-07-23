use rand::prelude::*;
use rand::rngs::SmallRng;

#[derive(Clone, Copy, PartialEq)]
pub enum InitialCondition {
    AllUp,
    AllDown,
    Random,
    Instanton,
}

pub enum BoundaryCondition {
    Periodic,
    Shifted,
}

pub struct IsingModel {
    pub l: usize,
    pub lattice: Vec<i32>,
    pub energy: i32,
    pub total_spin: i32,
    pub neighbors: Vec<[usize; 4]>,
}

impl IsingModel {
    pub fn new(l: usize, ic: InitialCondition, bc: BoundaryCondition) -> Self { 
        let mut rng: SmallRng = rand::make_rng();

        let mut lattice = Vec::with_capacity(l * l);
        for _i in 0..l {
            for j in 0..l {
                let val = match ic {
                    InitialCondition::AllUp => 1,
                    InitialCondition::AllDown => -1,
                    InitialCondition::Random => if rng.random::<bool>() { 1 } else { -1 },
                    InitialCondition::Instanton => if j < l / 2 { -1 } else { 1 },
                };
                lattice.push(val);
            }
        }

        let mut neighbors = Vec::with_capacity(l * l);
        for i in 0..l {
            for j in 0..l {
                let idx_top = if i == 0 { (l - 1) * l + j } else { (i - 1) * l + j };
                let idx_bottom = if i == l - 1 { j } else { (i + 1) * l + j };
                
                let idx_right = if j == l - 1 {
                    match bc {
                        BoundaryCondition::Periodic => i * l,
                        BoundaryCondition::Shifted => {
                            let shifted_i = (i + l - l / 4) % l;
                            shifted_i * l + j
                        }
                    }
                } else {
                    i * l + j + 1
                };
                
                let idx_left = if j == 0 {
                    match bc {
                        BoundaryCondition::Periodic => (i + 1) * l - 1,
                        BoundaryCondition::Shifted => {
                            let shifted_i = (i + l - l / 4) % l;
                            shifted_i * l + j
                        }
                    }
                } else {
                    i * l + j - 1
                };
                
                neighbors.push([idx_top, idx_right, idx_bottom, idx_left]);
            }
        }

        let mut model = Self {
            l,
            lattice,
            energy: 0,
            total_spin: 0,
            neighbors,
        };

        model.energy = model.compute_energy_int();
        model.total_spin = model.compute_magnetization();

        return model;
    }

    pub fn compute_energy_int(&self) -> i32 {
        let mut energy = 0;
        for idx in 0..self.l * self.l {
            let idx_right = self.neighbors[idx][1];
            let idx_bottom = self.neighbors[idx][2];
            energy -= self.lattice[idx] * (self.lattice[idx_right] + self.lattice[idx_bottom]);
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

    pub fn flip_energy_delta(&self, idx: usize) -> i32 {
        let sum_neighbors = self.lattice[self.neighbors[idx][0]]
            + self.lattice[self.neighbors[idx][1]]
            + self.lattice[self.neighbors[idx][2]]
            + self.lattice[self.neighbors[idx][3]];

        let alignment = self.lattice[idx] * sum_neighbors;

        return 2 * alignment;
    }

    pub fn flip(&mut self, idx: usize, energy_delta: i32) {
        self.lattice[idx] *= -1;
        self.energy += energy_delta;
        self.total_spin += 2 * self.lattice[idx];
    }

    pub fn swap_energy_delta(&self, idx1: usize, idx2: usize) -> i32 {
        // Assumes spins are adjacent and different!
        return self.flip_energy_delta(idx1) + self.flip_energy_delta(idx2) + 4;
    }

    pub fn swap(&mut self, idx1: usize, idx2: usize, energy_delta: i32) {
        // Assumes spins have different signs!
        self.lattice[idx1] *= -1;
        self.lattice[idx2] *= -1;
        self.energy += energy_delta;
    }
}
