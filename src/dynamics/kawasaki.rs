use crate::ising_model::IsingModel;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub struct KawasakiDynamics {
    pub beta: f64,
    pub m_plus: f64,
    pub current_h: Vec<i32>,
    
    swap_probabilities: [f64; 7],

    edges: Vec<[u32; 2]>,
    n_horizontal: usize,
    n_internal: usize,
    n_total: usize,

    rng: SmallRng,
}

impl KawasakiDynamics {
    pub fn new(l: usize, beta: f64, m_plus: f64) -> Self {
        let rng: SmallRng = rand::make_rng();
        let current_h = vec![0; l.saturating_sub(1)];

        let n_horizontal = l * (l - 1);
        let n_vertical = l * l;
        let n_reservoir = 2 * l;
        let n_internal = n_horizontal + n_vertical;
        let n_total = n_internal + n_reservoir;

        let mut edges = Vec::with_capacity(n_total);

        for y in 0..l {
            for x in 0..(l - 1) {
                let idx1 = (y * l + x) as u32;
                edges.push([idx1, idx1 + 1]);
            }
        }

        for y in 0..l {
            for x in 0..l {
                let idx1 = (y * l + x) as u32;
                let idx2 = if y == l - 1 { x as u32 } else { idx1 + l as u32 };
                edges.push([idx1, idx2]);
            }
        }

        for y in 0..l {
            edges.push([(y * l) as u32, 0]);
            edges.push([((y + 1) * l - 1) as u32, 1]);
        }

        let mut dyn_ = Self {
            beta,
            m_plus,
            current_h,
            swap_probabilities: [0.0; 7],
            edges,
            n_horizontal,
            n_internal,
            n_total,
            rng,
        };
        dyn_.precompute_swap_probabilities();
        dyn_
    }

    fn precompute_swap_probabilities(&mut self) {
        self.swap_probabilities = [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0];
        for i in 0..7 {
            self.swap_probabilities[i] =
                f64::exp(-self.beta * self.swap_probabilities[i]).clamp(0.0, 1.0);
        }
    }

    fn step(&mut self, model: &mut IsingModel) {
        let r = self.rng.random_range(0..self.n_total);
        let e = self.edges[r];
        let a = e[0] as usize;

        if r < self.n_internal {
            let b = e[1] as usize;

            if model.lattice[a] == model.lattice[b] {
                return;
            }

            let energy_delta = model.swap_energy_delta(a, b);
            let prob = self.swap_probabilities[(energy_delta / 4 + 3) as usize];

            if self.rng.random_bool(prob) {
                if r < self.n_horizontal {
                    let x = a % model.l;
                    self.current_h[x] += model.lattice[a];
                }
                model.swap(a, b, energy_delta);
            }
        } else {
            let sign = 1.0 - 2.0 * e[1] as f64;
            let flip_prob = (1.0 + sign * model.lattice[a] as f64 * self.m_plus) / 2.0;
            if self.rng.random_bool(flip_prob) {
                model.flip(a, model.flip_energy_delta(a));
            }
        }
    }

    pub fn sweep(&mut self, model: &mut IsingModel) {
        for _ in 0..self.n_total {
            self.step(model);
        }
    }
}
