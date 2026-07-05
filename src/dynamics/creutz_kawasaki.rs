use crate::ising_model::IsingModel;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub struct CreutzKawasakiDynamics {
    pub m: f64,
    pub demons_h: Vec<i32>,
    pub demons_v: Vec<i32>,
    pub current_h: Vec<i32>,

    rng: SmallRng,
}

impl CreutzKawasakiDynamics {
    pub fn new(l: usize, m: f64, starting_energy: i32) -> Self {
        let mut demons_h = Vec::new();
        let mut demons_v = Vec::new();
        for _ in 0..l * l {
            demons_h.push(starting_energy);
            demons_v.push(starting_energy);
        }

        let current_h = vec![0  ; l.saturating_sub(1)];

        let rng: SmallRng = rand::make_rng();

        return Self {
            m,
            demons_h,
            demons_v,
            current_h,
            rng,
        };
    }

    fn process_bonds(&mut self, model: &mut IsingModel) {
        // Horizontal even
        for y in 0..model.l {
            for x in (0..model.l - 1).step_by(2) {
                let idx1 = y * model.l + x;
                let idx2 = idx1 + 1;
                if model.lattice[idx1] == model.lattice[idx2] {
                    continue;
                }

                let energy_delta = model.swap_energy_delta(idx1, idx2);
                if energy_delta <= self.demons_h[idx1] {
                    let current = model.lattice[idx1];
                    model.swap(idx1, idx2, energy_delta);
                    self.demons_h[idx1] -= energy_delta;
                    self.current_h[x] += current;
                }
            }
        }
        // Horizontal odd
        for y in 0..model.l {
            for x in (1..model.l - 1).step_by(2) {
                let idx1 = y * model.l + x;
                let idx2 = idx1 + 1;
                if model.lattice[idx1] == model.lattice[idx2] {
                    continue;
                }

                let energy_delta = model.swap_energy_delta(idx1, idx2);
                if energy_delta <= self.demons_h[idx1] {
                    let current = model.lattice[idx1];
                    model.swap(idx1, idx2, energy_delta);
                    self.demons_h[idx1] -= energy_delta;
                    self.current_h[x] += current;
                }
            }
        }
        // Vertical even
        for y in (0..model.l).step_by(2) {
            for x in 0..model.l {
                let idx1 = y * model.l + x;
                let idx2 = if y == model.l - 1 { x } else { idx1 + model.l };
                if model.lattice[idx1] == model.lattice[idx2] {
                    continue;
                }

                let energy_delta = model.swap_energy_delta(idx1, idx2);
                if energy_delta <= self.demons_v[idx1] {
                    model.swap(idx1, idx2, energy_delta);
                    self.demons_v[idx1] -= energy_delta;
                }
            }
        }
        // Vertical odd
        for y in (1..model.l).step_by(2) {
            for x in 0..model.l {
                let idx1 = y * model.l + x;
                let idx2 = if y == model.l - 1 { x } else { idx1 + model.l };
                if model.lattice[idx1] == model.lattice[idx2] {
                    continue;
                }

                let energy_delta = model.swap_energy_delta(idx1, idx2);
                if energy_delta <= self.demons_v[idx1] {
                    model.swap(idx1, idx2, energy_delta);
                    self.demons_v[idx1] -= energy_delta;
                }
            }
        }
    }

    fn process_reservoirs(&mut self, model: &mut IsingModel) {
        for i in 0..model.l {
            let idx_l = i * model.l;
            let idx_r = (i + 1) * model.l - 1;

            let flipped_l = self
                .rng
                .random_bool((1.0 + model.lattice[idx_l] as f64 * self.m) / 2.0);
            let flipped_r = self
                .rng
                .random_bool((1.0 - model.lattice[idx_r] as f64 * self.m) / 2.0);

            if flipped_l {
                model.flip(idx_l, model.flip_energy_delta(idx_l));
            }
            if flipped_r {
                model.flip(idx_r, model.flip_energy_delta(idx_r));
            }
        }
    }

    pub fn sweep(&mut self, model: &mut IsingModel) {
        self.process_bonds(model);
        self.process_reservoirs(model);
    }
}
