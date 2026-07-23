use crate::ising_model::IsingModel;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub struct KawasakiReservoirStructDynamics {
    pub beta: f64,
    pub m_plus: f64,
    pub current_h: Vec<i32>,

    // Precomputed swap acceptance probabilities for ΔH ∈ {-12, -8, -4, 0, 4, 8, 12}.
    // Index = (ΔH / 4 + 3), so indices 0..7 map to ΔH = -12, -8, ..., 12.
    swap_probabilities: [f64; 7],

    // Compact precomputed edge table.
    // Internal bonds: [idx1, idx2]. Reservoir bonds: [idx, side].
    edges: Vec<[u32; 2]>,
    n_horizontal: usize,
    n_internal: usize, // n_horizontal + n_vertical
    n_total: usize,

    // Reservoir spin arrays: shuffled sequences of +1/-1 in the proportion
    // dictated by m_plus (right) and m_minus = -m_plus (left).
    // Left reservoir has magnetization -m_plus, right has +m_plus.
    reservoir_left: Vec<i32>,
    reservoir_right: Vec<i32>,
    cursor_left: usize,
    cursor_right: usize,

    rng: SmallRng,
}

impl KawasakiReservoirStructDynamics {
    pub fn new(l: usize, beta: f64, m_plus: f64, reservoir_size: usize) -> Self {
        let mut rng: SmallRng = rand::make_rng();
        let current_h = vec![0; l.saturating_sub(1)];

        let n_horizontal = l * (l - 1);
        let n_vertical = l * l; // includes periodic vertical wrapping
        let n_reservoir = 2 * l;
        let n_internal = n_horizontal + n_vertical;
        let n_total = n_internal + n_reservoir;

        let mut edges = Vec::with_capacity(n_total);

        // Horizontal internal bonds (x, x+1) for x in 0..L-1
        for y in 0..l {
            for x in 0..(l - 1) {
                let idx1 = (y * l + x) as u32;
                edges.push([idx1, idx1 + 1]);
            }
        }

        // Vertical internal bonds (y, y+1) with periodic wrapping
        for y in 0..l {
            for x in 0..l {
                let idx1 = (y * l + x) as u32;
                let idx2 = if y == l - 1 { x as u32 } else { idx1 + l as u32 };
                edges.push([idx1, idx2]);
            }
        }

        // Reservoir bonds: [site_idx, side] (side: 0 = left, 1 = right)
        for y in 0..l {
            edges.push([(y * l) as u32, 0]);
            edges.push([((y + 1) * l - 1) as u32, 1]);
        }

        // Build reservoir arrays.
        // Right reservoir: magnetization = m_plus, so n_up = N*(1+m)/2.
        // Left reservoir:  magnetization = -m_plus, so n_up = N*(1-m)/2.
        let reservoir_right = Self::build_reservoir(reservoir_size, m_plus, &mut rng);
        let reservoir_left = Self::build_reservoir(reservoir_size, -m_plus, &mut rng);

        let mut dyn_ = Self {
            beta,
            m_plus,
            current_h,
            swap_probabilities: [0.0; 7],
            edges,
            n_horizontal,
            n_internal,
            n_total,
            reservoir_left,
            reservoir_right,
            cursor_left: 0,
            cursor_right: 0,
            rng,
        };
        dyn_.precompute_swap_probabilities();
        dyn_
    }

    /// Build a shuffled array of `size` spins with the given magnetization.
    /// magnetization ∈ [-1, 1]: fraction of +1 spins = (1 + mag) / 2.
    fn build_reservoir(size: usize, magnetization: f64, rng: &mut SmallRng) -> Vec<i32> {
        let n_up = ((size as f64) * (1.0 + magnetization) / 2.0).round() as usize;
        let n_up = n_up.min(size);
        let mut arr = Vec::with_capacity(size);
        for _ in 0..n_up {
            arr.push(1);
        }
        for _ in n_up..size {
            arr.push(-1);
        }
        arr.shuffle(rng);
        arr
    }

    fn precompute_swap_probabilities(&mut self) {
        // c(i,j) = 1 if ΔH <= 0, else exp(-β * ΔH)
        for (k, dh) in [-12, -8, -4, 0, 4, 8, 12].iter().enumerate() {
            let dh = *dh as f64;
            self.swap_probabilities[k] = if dh <= 0.0 {
                1.0
            } else {
                (-self.beta * dh).exp()
            };
        }
    }

    fn step(&mut self, model: &mut IsingModel) {
        let r = self.rng.random_range(0..self.n_total);
        let e = self.edges[r];
        let a = e[0] as usize;

        if r < self.n_internal {
            // Internal bond: Kawasaki spin exchange with Metropolis acceptance
            let b = e[1] as usize;

            if model.lattice[a] == model.lattice[b] {
                return;
            }

            let energy_delta = model.swap_energy_delta(a, b);
            let prob = self.swap_probabilities[(energy_delta / 4 + 3) as usize];

            if self.rng.random_bool(prob) {
                // Track horizontal current before swapping
                if r < self.n_horizontal {
                    let x = a % model.l;
                    self.current_h[x] += model.lattice[a];
                }
                model.swap(a, b, energy_delta);
            }
        } else {
            // Reservoir bond: deterministic spin replacement from reservoir array.
            // Read the next spin from the corresponding reservoir and set the
            // boundary site to that value (if different, flip it).
            let side = e[1];

            let reservoir_spin = if side == 0 {
                let s = self.reservoir_left[self.cursor_left];
                self.cursor_left = (self.cursor_left + 1) % self.reservoir_left.len();
                s
            } else {
                let s = self.reservoir_right[self.cursor_right];
                self.cursor_right = (self.cursor_right + 1) % self.reservoir_right.len();
                s
            };

            if model.lattice[a] != reservoir_spin {
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

