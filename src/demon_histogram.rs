pub struct PerBondDemonHistogram {
    counts: Vec<u64>, // [n_bonds * max_levels], row-major
    n_bonds: usize,
    max_levels: usize,
}

impl PerBondDemonHistogram {
    pub fn new(n_bonds: usize, max_levels: usize) -> Self {
        Self {
            counts: vec![0; n_bonds * max_levels],
            n_bonds,
            max_levels,
        }
    }

    pub fn record(&mut self, bond_idx: usize, demon_energy: i32) {
        let k = (demon_energy / 4) as usize;
        if bond_idx < self.n_bonds && k < self.max_levels {
            self.counts[bond_idx * self.max_levels + k] += 1;
        }
    }

    pub fn record_all(&mut self, demons_h: &[i32], demons_v: &[i32], l: usize) {
        // Record horizontal demons: bond index = flat lattice index,
        // but skip the last column (x == l-1) since those aren't real bonds.
        for y in 0..l {
            for x in 0..(l - 1) {
                let idx = y * l + x;
                self.record(idx, demons_h[idx]);
            }
        }
        // Record vertical demons: bond index offset by l*l
        let offset = l * l;
        for i in 0..(l * l) {
            self.record(offset + i, demons_v[i]);
        }
    }

    pub fn record_creutz(&mut self, demons: &[i32]) {
        for (i, &e) in demons.iter().enumerate() {
            self.record(i, 4 * e);
        }
    }

    pub fn aggregate_betas(&self) -> AggregatedBetas {
        let mut mean_points = Vec::new();
        let mut std_upper_points = Vec::new();
        let mut std_lower_points = Vec::new();
        let mut min_points = Vec::new();
        let mut max_points = Vec::new();

        for i in 1..self.max_levels {
            let mut betas: Vec<f64> = Vec::new();

            for b in 0..self.n_bonds {
                let row = b * self.max_levels;
                let count_prev = self.counts[row + i - 1];
                let count_curr = self.counts[row + i];

                if count_prev >= 10 && count_curr >= 1 {
                    let ratio = count_curr as f64 / count_prev as f64;
                    betas.push(-ratio.ln());
                }
            }

            if betas.len() < 2 {
                continue;
            }

            let n = betas.len() as f64;
            let mean = betas.iter().sum::<f64>() / n;
            let variance = betas.iter().map(|&b| (b - mean) * (b - mean)).sum::<f64>() / n;
            let std = variance.sqrt();
            let min = betas.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = betas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let x = i as f64;
            mean_points.push([x, mean]);
            std_upper_points.push([x, mean + std]);
            std_lower_points.push([x, mean - std]);
            min_points.push([x, min]);
            max_points.push([x, max]);
        }

        AggregatedBetas {
            mean: mean_points,
            std_upper: std_upper_points,
            std_lower: std_lower_points,
            min: min_points,
            max: max_points,
        }
    }

    pub fn reset(&mut self) {
        self.counts.fill(0);
    }

    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|&c| c == 0)
    }
}

pub struct AggregatedBetas {
    pub mean: Vec<[f64; 2]>,
    pub std_upper: Vec<[f64; 2]>,
    pub std_lower: Vec<[f64; 2]>,
    pub min: Vec<[f64; 2]>,
    pub max: Vec<[f64; 2]>,
}
