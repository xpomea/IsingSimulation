use crate::ising_model::IsingModel;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub struct CreutzKawasakiRandomDynamics {
    pub m: f64,
    pub demons_h: Vec<i32>,
    pub demons_v: Vec<i32>,
    pub current_h: Vec<i32>,

    // Compact precomputed edge table.
    // Internal bonds: [idx1, idx2]. Reservoir bonds: [idx, side].
    // 8 bytes per edge (~160KB for L=100, fits in L2).
    edges: Vec<[u32; 2]>,
    n_horizontal: usize,
    n_internal: usize, // n_horizontal + n_vertical
    n_total: usize,

    rng: SmallRng,
}

impl CreutzKawasakiRandomDynamics {
    pub fn new(l: usize, m: f64, starting_energy: i32) -> Self {
        let n = l * l;
        let demons_h = vec![starting_energy; n];
        let demons_v = vec![starting_energy; n];
        let current_h = vec![0; l.saturating_sub(1)];

        let rng: SmallRng = rand::make_rng();

        let n_horizontal = l * (l - 1);
        let n_vertical = l * l;
        let n_reservoir = 2 * l;
        let n_internal = n_horizontal + n_vertical;
        let n_total = n_internal + n_reservoir;

        let mut edges = Vec::with_capacity(n_total);

        // Horizontal internal bonds
        for y in 0..l {
            for x in 0..(l - 1) {
                let idx1 = (y * l + x) as u32;
                edges.push([idx1, idx1 + 1]);
            }
        }

        // Vertical internal bonds
        for y in 0..l {
            for x in 0..l {
                let idx1 = (y * l + x) as u32;
                let idx2 = if y == l - 1 { x as u32 } else { idx1 + l as u32 };
                edges.push([idx1, idx2]);
            }
        }

        // Reservoir bonds: [site_idx, side]  (side: 0 = left/+m, 1 = right/-m)
        for y in 0..l {
            edges.push([(y * l) as u32, 0]);
            edges.push([((y + 1) * l - 1) as u32, 1]);
        }

        Self {
            m,
            demons_h,
            demons_v,
            current_h,
            edges,
            n_horizontal,
            n_internal,
            n_total,
            rng,
        }
    }

    pub fn sweep(&mut self, model: &mut IsingModel) {
        let n_total = self.n_total;
        let n_horizontal = self.n_horizontal;
        let n_internal = self.n_internal;

        for _ in 0..n_total {
            let r = self.rng.random_range(0..n_total);
            let e = self.edges[r];
            let a = e[0] as usize;

            if r < n_internal {
                // Internal bond (horizontal or vertical)
                let b = e[1] as usize;

                if model.lattice[a] == model.lattice[b] {
                    continue;
                }

                let energy_delta = model.swap_energy_delta(a, b);

                if r < n_horizontal {
                    // Horizontal
                    if energy_delta <= self.demons_h[a] {
                        let x = a % model.l;
                        self.current_h[x] += model.lattice[a];
                        model.swap(a, b, energy_delta);
                        self.demons_h[a] -= energy_delta;
                    }
                } else {
                    // Vertical
                    if energy_delta <= self.demons_v[a] {
                        model.swap(a, b, energy_delta);
                        self.demons_v[a] -= energy_delta;
                    }
                }
            } else {
                // Reservoir bond
                let sign = 1.0 - 2.0 * e[1] as f64;
                let flip_prob = (1.0 + sign * model.lattice[a] as f64 * self.m) / 2.0;
                if self.rng.random_bool(flip_prob) {
                    model.flip(a, model.flip_energy_delta(a));
                }
            }
        }
    }
}
