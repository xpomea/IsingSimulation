use crate::dynamics::{BondSelection, ReservoirType};
use crate::ising_model::IsingModel;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub struct CreutzKawasakiDynamics {
    pub m: f64,
    pub demons_h: Vec<i32>,
    pub demons_v: Vec<i32>,
    pub current_h: Vec<i32>,

    bond_selection: BondSelection,
    reservoir_type: ReservoirType,

    edges: Vec<[u32; 2]>,
    n_horizontal: usize,
    n_internal: usize,
    n_total: usize,

    reservoir_left: Vec<i32>,
    reservoir_right: Vec<i32>,
    cursor_left: usize,
    cursor_right: usize,

    rng: SmallRng,
}

impl CreutzKawasakiDynamics {
    pub fn new(
        l: usize,
        m: f64,
        starting_energy: i32,
        bond_selection: BondSelection,
        reservoir_type: ReservoirType,
    ) -> Self {
        let mut rng: SmallRng = rand::make_rng();
        let demons_h = vec![starting_energy; l * l];
        let demons_v = vec![starting_energy; l * l];
        let current_h = vec![0; l.saturating_sub(1)];

        let n_horizontal = l * (l - 1);
        let n_vertical = l * l;
        let n_reservoir = 2 * l;
        let n_internal = n_horizontal + n_vertical;
        let n_total = n_internal + n_reservoir;

        let mut edges = Vec::new();
        if bond_selection == BondSelection::Random {
            edges.reserve(n_total);

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
        }

        let (reservoir_left, reservoir_right) = if reservoir_type == ReservoirType::Quenched {
            let reservoir_size = 10000;
            (
                Self::build_reservoir(reservoir_size, -m, &mut rng),
                Self::build_reservoir(reservoir_size, m, &mut rng),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        Self {
            m,
            demons_h,
            demons_v,
            current_h,
            bond_selection,
            reservoir_type,
            edges,
            n_horizontal,
            n_internal,
            n_total,
            reservoir_left,
            reservoir_right,
            cursor_left: 0,
            cursor_right: 0,
            rng,
        }
    }

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

            if r < self.n_horizontal {
                if energy_delta <= self.demons_h[a] {
                    let x = a % model.l;
                    self.current_h[x] += model.lattice[a];
                    model.swap(a, b, energy_delta);
                    self.demons_h[a] -= energy_delta;
                }
            } else {
                if energy_delta <= self.demons_v[a] {
                    model.swap(a, b, energy_delta);
                    self.demons_v[a] -= energy_delta;
                }
            }
        } else {
            match self.reservoir_type {
                ReservoirType::Annealed => {
                    let sign = 1.0 - 2.0 * e[1] as f64;
                    let flip_prob = (1.0 + sign * model.lattice[a] as f64 * self.m) / 2.0;
                    if self.rng.random_bool(flip_prob) {
                        model.flip(a, model.flip_energy_delta(a));
                    }
                }
                ReservoirType::Quenched => {
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
        }
    }

    fn process_bonds_checkerboard(&mut self, model: &mut IsingModel) {
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

            match self.reservoir_type {
                ReservoirType::Annealed => {
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
                ReservoirType::Quenched => {
                    let s_l = self.reservoir_left[self.cursor_left];
                    self.cursor_left = (self.cursor_left + 1) % self.reservoir_left.len();
                    let s_r = self.reservoir_right[self.cursor_right];
                    self.cursor_right = (self.cursor_right + 1) % self.reservoir_right.len();

                    if model.lattice[idx_l] != s_l {
                        model.flip(idx_l, model.flip_energy_delta(idx_l));
                    }
                    if model.lattice[idx_r] != s_r {
                        model.flip(idx_r, model.flip_energy_delta(idx_r));
                    }
                }
            }
        }
    }

    pub fn sweep(&mut self, model: &mut IsingModel) {
        match self.bond_selection {
            BondSelection::Checkerboard => {
                self.process_bonds_checkerboard(model);
                self.process_reservoirs(model);
            }
            BondSelection::Random => {
                for _ in 0..self.n_total {
                    self.step(model);
                }
            }
        }
    }
}
