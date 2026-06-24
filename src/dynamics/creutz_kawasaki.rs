use crate::ising_model::IsingModel;

pub struct Bond {
    pub idx1: usize,
    pub idx2: usize,
    pub demon_energy: i32,
}

pub struct CreutzKawasakiDynamics {
    pub m: f64,
    pub demons: Vec<i32>,
}

impl CreutzKawasakiDynamics {
    pub fn new(l: usize, m: f64) -> Self {
        
        return Self {
            m,
            demons: Vec::new(),
        }
    }
    
    fn process_bonds(&mut self, group: usize, model: &mut IsingModel) {
            
    }

    fn process_reservoirs(&mut self, model: &mut IsingModel) {
        
    }

    pub fn sweep(&mut self, model: &mut IsingModel) {
        for g in 0..4 {
            self.process_bonds(g, model);
        }
        self.process_reservoirs(model);
    }
}
