use crate::dynamics::common::{
    build_edges, build_reservoir, process_reservoir_bond, process_reservoirs,
};
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
    }
}
