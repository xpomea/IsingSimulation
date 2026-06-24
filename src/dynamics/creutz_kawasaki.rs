use crate::ising_model::IsingModel;

pub struct CreutzKawasakiDynamics {
    pub m: f64,
    pub demons_h: Vec<i32>,
    pub demons_v: Vec<i32>,
}

impl CreutzKawasakiDynamics {
    pub fn new(l: usize, m: f64, starting_energy: i32) -> Self {
        let mut demons_h = Vec::new();
        let mut demons_v = Vec::new();
        for i in 0..l * l {
            demons_h.push(starting_energy);
            demons_v.push(starting_energy);
        }

        return Self { m, demons_h, demons_v };
    }

    fn process_bonds(&mut self, model: &mut IsingModel) {
        for y in 0..model.l {
            for x in (0..model.l - 1).step_by(2) {
                let idx1 = y * model.l + x;
                let idx2 = idx1 + 1;

            }
        }
        
        for y in 0..model.l {
            for x in (1..model.l - 1).step_by(2) {
                let idx1 = y * model.l + x;
                let idx2 = idx1 + 1;
                
            }
        }
        
        for y in (0..model.l).step_by(2) {
            for x in 0..model.l {
                let idx1 = y * model.l + x;
                let idx2 = if y == model.l - 1 { x } else { idx1 + model.l };
                
            }
        }
        
        for y in (1..model.l).step_by(2) {
            for x in 0..model.l {
                let idx1 = y * model.l + x;
                let idx2 = if y == model.l - 1 { x } else { idx1 + model.l };
                
            }
        }
    }

    fn process_reservoirs(&mut self, model: &mut IsingModel) {}

    pub fn sweep(&mut self, model: &mut IsingModel) {
        self.process_bonds(model);
        self.process_reservoirs(model);
    }
}
