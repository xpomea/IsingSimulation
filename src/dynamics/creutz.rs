use crate::ising_model::IsingModel;

pub struct CreutzDynamics {
    pub demons: Vec<i32>,
    pub max_demon: i32,
}

impl CreutzDynamics {
    pub fn new(l: usize, initial_demon_energy: i32, max_demon: i32) -> Self {
        let demons = vec![initial_demon_energy; l * l];
        Self { demons, max_demon }
    }

    fn half_sweep(&mut self, model: &mut IsingModel, parity: usize) {
        let l = model.l;
        for y in 0..l {
            for x in 0..l {
                if (x + y) % 2 != parity {
                    continue;
                }
                let idx = y * l + x;
                let delta_h = model.flip_energy_delta(idx);
                let new_demon = self.demons[idx] - delta_h / 4;
                if new_demon >= 0 && new_demon <= self.max_demon {
                    model.flip(idx, delta_h);
                    self.demons[idx] = new_demon;
                }
            }
        }
    }

    pub fn sweep(&mut self, model: &mut IsingModel) {
        self.half_sweep(model, 0);
        self.half_sweep(model, 1);
    }
}
