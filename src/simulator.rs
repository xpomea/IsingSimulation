use crate::dynamics::{BondSelection, CreutzThermalDynamics, DemonReplacementMode, ReservoirType};
use crate::ising_model::{BoundaryCondition, InitialCondition, IsingModel};
use std::time::Instant;

pub const BETA_C: f64 = 0.44068679350977147; // 0.5 * ln(1 + sqrt(2))

/// Spontaneous magnetization m_beta = [1 - sinh^(-4)(2*beta)]^(1/8) for 2D square lattice Ising model
pub fn spontaneous_magnetization(beta: f64) -> f64 {
    if beta <= BETA_C {
        0.0
    } else {
        let sinh_2b = (2.0 * beta).sinh();
        let term = sinh_2b.powi(-4);
        if term >= 1.0 {
            0.0
        } else {
            (1.0 - term).powf(0.125)
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub l: usize,
    pub beta: f64,
    pub m_plus: f64,
    pub therm_sweeps: usize,
    pub meas_sweeps: usize,
    pub replacement_mode: DemonReplacementMode,
    pub bond_selection: BondSelection,
    pub reservoir_type: ReservoirType,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            l: 40,
            beta: 1.0,
            m_plus: 0.999,
            therm_sweeps: 1_000_000,
            meas_sweeps: 500_000,
            replacement_mode: DemonReplacementMode::PerStep,
            bond_selection: BondSelection::Random,
            reservoir_type: ReservoirType::Annealed,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimResult {
    pub beta: f64,
    pub m_plus: f64,
    pub l: usize,
    pub therm_sweeps: usize,
    pub meas_sweeps: usize,
    pub sum_current: i64,
    pub j_flux: f64,
    pub j_per_sweep: f64,
    pub final_mag: f64,
    pub mean_mag: f64,
    pub final_energy_per_spin: f64,
    pub zero_crossings: usize,
    pub phase: String,
    pub is_metastable: bool,
    pub is_uphill: bool,
    pub elapsed_sec: f64,
    pub col_mag: Vec<f64>,
}

pub fn run_simulation(config: &SimConfig) -> SimResult {
    let start = Instant::now();

    let mut model =
        IsingModel::new(config.l, InitialCondition::Instanton, BoundaryCondition::Shifted);
    let mut dynamics = CreutzThermalDynamics::new(
        config.l,
        config.m_plus,
        config.beta,
        config.replacement_mode,
        config.bond_selection,
        config.reservoir_type,
    );

    // Thermalization / burn-in phase
    for _ in 0..config.therm_sweeps {
        dynamics.sweep(&mut model);
    }

    // Reset current counters for stationary measurement
    dynamics.current_h.fill(0);

    // Measurement phase with column magnetization accumulation
    let mut col_spin_sum = vec![0i64; config.l];

    for _ in 0..config.meas_sweeps {
        dynamics.sweep(&mut model);
        for x in 0..config.l {
            let mut col_sum = 0i64;
            for y in 0..config.l {
                col_sum += model.lattice[y * config.l + x] as i64;
            }
            col_spin_sum[x] += col_sum;
        }
    }

    let sum_current: i64 = dynamics.current_h.iter().map(|&c| c as i64).sum();
    let num_columns = config.l.saturating_sub(1).max(1);
    let total_samples = (num_columns * config.meas_sweeps).max(1) as f64;
    let j_flux = sum_current as f64 / total_samples;
    let j_per_sweep = if config.meas_sweeps > 0 {
        sum_current as f64 / config.meas_sweeps as f64
    } else {
        0.0
    };

    let col_denom = (config.l * config.meas_sweeps).max(1) as f64;
    let col_mag: Vec<f64> = col_spin_sum.iter().map(|&s| s as f64 / col_denom).collect();
    let mean_mag: f64 = col_mag.iter().sum::<f64>() / config.l as f64;

    // Count zero-crossings in time-averaged profile with hysteresis
    let mut zero_crossings = 0;
    let mut prev_sign = 0i32;
    for &m_val in &col_mag {
        let sign = if m_val > 0.05 {
            1
        } else if m_val < -0.05 {
            -1
        } else {
            0
        };
        if sign != 0 {
            if prev_sign != 0 && sign != prev_sign {
                zero_crossings += 1;
            }
            prev_sign = sign;
        }
    }

    // Phase identification:
    // - Stable: instanton intact (single domain wall), downhill current (J <= 0)
    // - Metastable: single bump, bulk uniformly ordered (|mean_mag| >= 0.4), uphill current (J > 0)
    // - Weakly-unstable: double bump / multiple domain walls (zero_crossings >= 2 or broken bump)
    // - Chaotic: disordered or high fluctuations
    let (phase, is_metastable) = if j_flux <= 0.0 && zero_crossings <= 1 && mean_mag.abs() < 0.4 {
        ("stable".to_string(), false)
    } else if j_flux > 0.0 && zero_crossings <= 1 && mean_mag.abs() >= 0.4 {
        ("metastable".to_string(), true)
    } else if zero_crossings >= 2 || (j_flux > 0.0 && mean_mag.abs() < 0.4) {
        ("weakly_unstable".to_string(), false)
    } else {
        ("chaotic".to_string(), false)
    };

    let elapsed_sec = start.elapsed().as_secs_f64();
    let final_mag = model.magnetization();
    let final_energy_per_spin = model.energy as f64 / (config.l * config.l) as f64;
    let is_uphill = j_flux > 0.0;

    SimResult {
        beta: config.beta,
        m_plus: config.m_plus,
        l: config.l,
        therm_sweeps: config.therm_sweeps,
        meas_sweeps: config.meas_sweeps,
        sum_current,
        j_flux,
        j_per_sweep,
        final_mag,
        mean_mag,
        final_energy_per_spin,
        zero_crossings,
        phase,
        is_metastable,
        is_uphill,
        elapsed_sec,
        col_mag,
    }
}
