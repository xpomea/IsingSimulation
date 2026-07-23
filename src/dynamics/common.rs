use crate::dynamics::ReservoirType;
use crate::ising_model::IsingModel;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub fn build_edges(l: usize) -> (Vec<[u32; 2]>, usize, usize, usize) {
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

    (edges, n_horizontal, n_internal, n_total)
}

pub fn build_reservoir(size: usize, magnetization: f64, rng: &mut SmallRng) -> Vec<i32> {
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

pub fn process_reservoirs(
    model: &mut IsingModel,
    reservoir_type: ReservoirType,
    m: f64,
    reservoir_left: &[i32],
    reservoir_right: &[i32],
    cursor_left: &mut usize,
    cursor_right: &mut usize,
    rng: &mut SmallRng,
) {
    for i in 0..model.l {
        let idx_l = i * model.l;
        let idx_r = (i + 1) * model.l - 1;

        match reservoir_type {
            ReservoirType::Annealed => {
                let flipped_l = rng.random_bool((1.0 + model.lattice[idx_l] as f64 * m) / 2.0);
                let flipped_r = rng.random_bool((1.0 - model.lattice[idx_r] as f64 * m) / 2.0);

                if flipped_l {
                    model.flip(idx_l, model.flip_energy_delta(idx_l));
                }
                if flipped_r {
                    model.flip(idx_r, model.flip_energy_delta(idx_r));
                }
            }
            ReservoirType::Quenched => {
                let s_l = reservoir_left[*cursor_left];
                *cursor_left = (*cursor_left + 1) % reservoir_left.len();
                let s_r = reservoir_right[*cursor_right];
                *cursor_right = (*cursor_right + 1) % reservoir_right.len();

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

pub fn process_reservoir_bond(
    model: &mut IsingModel,
    site: usize,
    side: u32,
    reservoir_type: ReservoirType,
    m: f64,
    reservoir_left: &[i32],
    reservoir_right: &[i32],
    cursor_left: &mut usize,
    cursor_right: &mut usize,
    rng: &mut SmallRng,
) {
    match reservoir_type {
        ReservoirType::Annealed => {
            let sign = 1.0 - 2.0 * side as f64;
            let flip_prob = (1.0 + sign * model.lattice[site] as f64 * m) / 2.0;
            if rng.random_bool(flip_prob) {
                model.flip(site, model.flip_energy_delta(site));
            }
        }
        ReservoirType::Quenched => {
            let reservoir_spin = if side == 0 {
                let s = reservoir_left[*cursor_left];
                *cursor_left = (*cursor_left + 1) % reservoir_left.len();
                s
            } else {
                let s = reservoir_right[*cursor_right];
                *cursor_right = (*cursor_right + 1) % reservoir_right.len();
                s
            };
            if model.lattice[site] != reservoir_spin {
                model.flip(site, model.flip_energy_delta(site));
            }
        }
    }
}
