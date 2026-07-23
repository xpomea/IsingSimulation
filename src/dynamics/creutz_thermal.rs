use crate::dynamics::common::{
    build_edges, build_reservoir, process_reservoir_bond, process_reservoirs,
};
use crate::dynamics::{BondSelection, DemonReplacementMode, ReservoirType};
use crate::ising_model::IsingModel;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub struct CreutzThermalDynamics {
    pub m: f64,
    pub beta: f64,
    pub demons_h: Vec<i32>,
    pub demons_v: Vec<i32>,
    pub current_h: Vec<i32>,

    replacement_mode: DemonReplacementMode,
    bond_selection: BondSelection,
    reservoir_type: ReservoirType,

    demon_pool: Vec<i32>,
    demon_pool_cursor: usize,

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

impl CreutzThermalDynamics {
    pub fn new(
        l: usize,
        m: f64,
        beta: f64,
        replacement_mode: DemonReplacementMode,
        bond_selection: BondSelection,
        reservoir_type: ReservoirType,
    ) -> Self {
        let mut rng: SmallRng = rand::make_rng();
        let current_h = vec![0; l.saturating_sub(1)];

        let demon_pool = Self::build_demon_pool(100_000, beta, &mut rng);

        let demons_h: Vec<i32> = (0..l * l)
            .map(|i| demon_pool[i % demon_pool.len()])
            .collect();
        let demons_v: Vec<i32> = (0..l * l)
            .map(|i| demon_pool[(l * l + i) % demon_pool.len()])
            .collect();
        let demon_pool_cursor = (2 * l * l) % demon_pool.len();

        let (edges, n_horizontal, n_internal, n_total) = if bond_selection == BondSelection::Random
        {
            build_edges(l)
        } else {
            (Vec::new(), l * (l - 1), l * (l - 1) + l * l, 2 * l * l + 2 * l)
        };

        let (reservoir_left, reservoir_right) = if reservoir_type == ReservoirType::Quenched {
            let reservoir_size = 10000;
            (
                build_reservoir(reservoir_size, -m, &mut rng),
                build_reservoir(reservoir_size, m, &mut rng),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        Self {
            m,
            beta,
            demons_h,
            demons_v,
            current_h,
            replacement_mode,
            bond_selection,
            reservoir_type,
            demon_pool,
            demon_pool_cursor,
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

    fn build_demon_pool(size: usize, beta: f64, rng: &mut SmallRng) -> Vec<i32> {
        let p = (-4.0 * beta).exp();
        let mut pool = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = rng.random_range(0.0_f64..1.0_f64);
            let k = if p > 0.0 && p < 1.0 {
                (u.ln() / p.ln()).floor() as i32
            } else if p >= 1.0 {
                rng.random_range(0..100)
            } else {
                0
            };
            pool.push(4 * k.max(0));
        }
        pool.shuffle(rng);
        pool
    }

    #[inline]
    fn next_demon_from_pool(&mut self) -> i32 {
        let d = self.demon_pool[self.demon_pool_cursor];
        self.demon_pool_cursor = (self.demon_pool_cursor + 1) % self.demon_pool.len();
        d
    }

    #[inline]
    fn process_horizontal_bond(&mut self, model: &mut IsingModel, y: usize, x: usize) {
        let idx1 = y * model.l + x;
        let idx2 = idx1 + 1;
        if model.lattice[idx1] == model.lattice[idx2] {
            return;
        }

        let energy_delta = model.swap_energy_delta(idx1, idx2);
        if energy_delta <= self.demons_h[idx1] {
            let current = model.lattice[idx1];
            model.swap(idx1, idx2, energy_delta);
            self.demons_h[idx1] -= energy_delta;
            self.current_h[x] += current;
        }

        if self.replacement_mode == DemonReplacementMode::PerStep {
            let new_d = self.next_demon_from_pool();
            self.demons_h[idx1] = new_d;
        }
    }

    #[inline]
    fn process_vertical_bond(&mut self, model: &mut IsingModel, y: usize, x: usize) {
        let idx1 = y * model.l + x;
        let idx2 = if y == model.l - 1 { x } else { idx1 + model.l };
        if model.lattice[idx1] == model.lattice[idx2] {
            return;
        }

        let energy_delta = model.swap_energy_delta(idx1, idx2);
        if energy_delta <= self.demons_v[idx1] {
            model.swap(idx1, idx2, energy_delta);
            self.demons_v[idx1] -= energy_delta;
        }

        if self.replacement_mode == DemonReplacementMode::PerStep {
            let new_d = self.next_demon_from_pool();
            self.demons_v[idx1] = new_d;
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

            let is_horizontal = r < self.n_horizontal;

            if is_horizontal {
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

            if self.replacement_mode == DemonReplacementMode::PerStep {
                let new_demon = self.next_demon_from_pool();
                if is_horizontal {
                    self.demons_h[a] = new_demon;
                } else {
                    self.demons_v[a] = new_demon;
                }
            }
        } else {
            process_reservoir_bond(
                model,
                a,
                e[1],
                self.reservoir_type,
                self.m,
                &self.reservoir_left,
                &self.reservoir_right,
                &mut self.cursor_left,
                &mut self.cursor_right,
                &mut self.rng,
            );
        }
    }

    fn process_bonds_checkerboard(&mut self, model: &mut IsingModel) {
        for start_x in [0, 1] {
            for y in 0..model.l {
                for x in (start_x..model.l - 1).step_by(2) {
                    self.process_horizontal_bond(model, y, x);
                }
            }
        }
        for start_y in [0, 1] {
            for y in (start_y..model.l).step_by(2) {
                for x in 0..model.l {
                    self.process_vertical_bond(model, y, x);
                }
            }
        }
    }

    pub fn sweep(&mut self, model: &mut IsingModel) {
        match self.bond_selection {
            BondSelection::Checkerboard => {
                self.process_bonds_checkerboard(model);
                process_reservoirs(
                    model,
                    self.reservoir_type,
                    self.m,
                    &self.reservoir_left,
                    &self.reservoir_right,
                    &mut self.cursor_left,
                    &mut self.cursor_right,
                    &mut self.rng,
                );
            }
            BondSelection::Random => {
                for _ in 0..self.n_total {
                    self.step(model);
                }
            }
        }

        if self.replacement_mode == DemonReplacementMode::PerSweep {
            for i in 0..self.demons_h.len() {
                self.demons_h[i] = self.next_demon_from_pool();
            }
            for i in 0..self.demons_v.len() {
                self.demons_v[i] = self.next_demon_from_pool();
            }
        }
    }
}
