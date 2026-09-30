use IsingSimulation::dynamics::{BondSelection, DemonReplacementMode, ReservoirType};
use IsingSimulation::simulator::{run_simulation, spontaneous_magnetization, SimConfig, SimResult, BETA_C};
use rayon::prelude::*;
use std::collections::HashSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Debug)]
enum ScanMode {
    Band,
    Full,
    All,
}

struct CliArgs {
    mode: ScanMode,
    therm_sweeps: usize,
    meas_sweeps: usize,
    threads: usize,
    l: usize,
    output_path: PathBuf,
    beta_min: f64,
    beta_max: f64,
    beta_step: f64,
    m_min: f64,
    m_max: f64,
    m_step: f64,
    band_depth: f64,
    band_above: f64,
    band_m_step: f64,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            mode: ScanMode::Band,
            therm_sweeps: 3_000_000,
            meas_sweeps: 1_000_000,
            threads: 6,
            l: 40,
            output_path: PathBuf::from("results/creutz_thermal_scan.csv"),
            beta_min: 0.45,
            beta_max: 1.20,
            beta_step: 0.02,
            m_min: 0.0,
            m_max: 1.0,
            m_step: 0.05,
            band_depth: 0.03,
            band_above: 1.0,
            band_m_step: 0.0005,
        }
    }
}

fn print_help() {
    println!(
        r#"
IsingSimulation — Creutz Thermal Grid Scanner

USAGE:
    scan_grid [OPTIONS]

OPTIONS:
    --mode <band|full|all>      Scan mode (default: band)
                                'band': dense scan in the region below m_beta curve
                                'full': uniform 2D grid over [beta_min, beta_max] x [m_min, m_max]
                                'all':  combined full grid + dense band
    --therm <N>                 Thermalization / burn-in sweeps (default: 1000000)
    --meas <N>                  Measurement sweeps for current (default: 500000)
    --threads <N>               Number of CPU worker threads (default: 6)
    --l <N>                     Lattice size L (default: 40)
    --output <PATH>             CSV output path (default: results/creutz_thermal_scan.csv)

GRID PARAMETERS:
    --beta-min <VAL>            Minimum beta (default: 0.45 for band, 0.20 for full)
    --beta-max <VAL>            Maximum beta (default: 1.20)
    --beta-step <VAL>           Step size for beta (default: 0.02)
    --m-min <VAL>               Minimum m+ for full mode (default: 0.0)
    --m-max <VAL>               Maximum m+ for full mode (default: 1.0)
    --m-step <VAL>              Step size for m+ in full mode (default: 0.05)
    --band-depth <VAL>          Distance below m_beta to explore in band mode (default: 0.15)
    --band-above <VAL>          Distance above m_beta to explore in band mode (default: 0.01)
    --band-m-step <VAL>         Step size for m+ in band mode (default: 0.002)

    -h, --help                  Print this help message
"#
    );
}

fn parse_cli_args() -> CliArgs {
    let mut args = CliArgs::default();
    let raw: Vec<String> = env::args().skip(1).collect();

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--mode" => {
                i += 1;
                if i < raw.len() {
                    args.mode = match raw[i].to_lowercase().as_str() {
                        "full" => {
                            args.beta_min = 0.20;
                            ScanMode::Full
                        }
                        "all" => {
                            args.beta_min = 0.20;
                            ScanMode::All
                        }
                        _ => ScanMode::Band,
                    };
                }
            }
            "--therm" => {
                i += 1;
                if i < raw.len() {
                    args.therm_sweeps = raw[i].parse().unwrap_or(args.therm_sweeps);
                }
            }
            "--meas" => {
                i += 1;
                if i < raw.len() {
                    args.meas_sweeps = raw[i].parse().unwrap_or(args.meas_sweeps);
                }
            }
            "--threads" => {
                i += 1;
                if i < raw.len() {
                    args.threads = raw[i].parse().unwrap_or(args.threads);
                }
            }
            "--l" => {
                i += 1;
                if i < raw.len() {
                    args.l = raw[i].parse().unwrap_or(args.l);
                }
            }
            "--output" => {
                i += 1;
                if i < raw.len() {
                    args.output_path = PathBuf::from(&raw[i]);
                }
            }
            "--beta-min" => {
                i += 1;
                if i < raw.len() {
                    args.beta_min = raw[i].parse().unwrap_or(args.beta_min);
                }
            }
            "--beta-max" => {
                i += 1;
                if i < raw.len() {
                    args.beta_max = raw[i].parse().unwrap_or(args.beta_max);
                }
            }
            "--beta-step" => {
                i += 1;
                if i < raw.len() {
                    args.beta_step = raw[i].parse().unwrap_or(args.beta_step);
                }
            }
            "--m-min" => {
                i += 1;
                if i < raw.len() {
                    args.m_min = raw[i].parse().unwrap_or(args.m_min);
                }
            }
            "--m-max" => {
                i += 1;
                if i < raw.len() {
                    args.m_max = raw[i].parse().unwrap_or(args.m_max);
                }
            }
            "--m-step" => {
                i += 1;
                if i < raw.len() {
                    args.m_step = raw[i].parse().unwrap_or(args.m_step);
                }
            }
            "--band-depth" => {
                i += 1;
                if i < raw.len() {
                    args.band_depth = raw[i].parse().unwrap_or(args.band_depth);
                }
            }
            "--band-above" => {
                i += 1;
                if i < raw.len() {
                    args.band_above = raw[i].parse().unwrap_or(args.band_above);
                }
            }
            "--band-m-step" => {
                i += 1;
                if i < raw.len() {
                    args.band_m_step = raw[i].parse().unwrap_or(args.band_m_step);
                }
            }
            other => {
                eprintln!("Unknown argument: {other}. Use --help to view available options.");
            }
        }
        i += 1;
    }

    args
}

fn point_key(beta: f64, m: f64) -> (i64, i64) {
    ((beta * 100_000.0).round() as i64, (m * 100_000.0).round() as i64)
}

fn read_existing_points(path: &Path) -> HashSet<(i64, i64)> {
    let mut set = HashSet::new();
    if !path.exists() {
        return set;
    }

    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            let line = line.trim();
            if line.is_empty() || line.starts_with("beta") {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                if let (Ok(beta), Ok(m)) = (parts[0].trim().parse::<f64>(), parts[1].trim().parse::<f64>()) {
                    set.insert(point_key(beta, m));
                }
            }
        }
    }
    set
}

fn generate_points(args: &CliArgs) -> Vec<(f64, f64)> {
    let mut points = Vec::new();
    let mut seen = HashSet::new();

    let mut add_point = |b: f64, m: f64, pts: &mut Vec<(f64, f64)>| {
        let b = (b * 10_000.0).round() / 10_000.0;
        let m = (m * 100_000.0).round() / 100_000.0;
        let key = point_key(b, m);
        if seen.insert(key) {
            pts.push((b, m));
        }
    };

    // Full 2D grid
    if args.mode == ScanMode::Full || args.mode == ScanMode::All {
        let mut b = args.beta_min;
        while b <= args.beta_max + 1e-6 {
            let mut m = args.m_min;
            while m <= args.m_max + 1e-6 {
                add_point(b, m, &mut points);
                m += args.m_step;
            }
            b += args.beta_step;
        }
    }

    // Dense band scan below m_beta
    if args.mode == ScanMode::Band || args.mode == ScanMode::All {
        let b_start = args.beta_min.max(BETA_C + 0.005);
        let mut b = b_start;
        while b <= args.beta_max + 1e-6 {
            let m_theo = spontaneous_magnetization(b);
            if m_theo > 0.0 {
                let m_start = (m_theo - args.band_depth).max(0.0);
                let m_end = (m_theo + args.band_above).min(1.0);

                let mut m = m_start;
                while m <= m_end + 1e-6 {
                    add_point(b, m, &mut points);
                    m += args.band_m_step;
                }
                add_point(b, m_theo, &mut points);
                add_point(b, 1.0, &mut points);
            }
            b += args.beta_step;
        }
    }

    points
}

fn main() {
    let args = parse_cli_args();

    println!("============================================================");
    println!("     Ising Model: Creutz-Thermal Uphill Current Scan        ");
    println!("============================================================");
    println!("Lattice size L:        {}", args.l);
    println!("Thermalization sweeps: {}", args.therm_sweeps);
    println!("Measurement sweeps:    {}", args.meas_sweeps);
    println!("Worker threads:        {}", args.threads);
    println!("Scan mode:             {:?}", args.mode);
    println!("Output CSV:            {}", args.output_path.display());
    println!("Beta range:            [{:.4}, {:.4}], step={:.4}", args.beta_min, args.beta_max, args.beta_step);
    println!("============================================================");

    // Initialize Rayon thread pool with requested thread count
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

    // Prepare output directory
    if let Some(parent) = args.output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("Failed to create results output directory");
        }
    }

    // Read already completed points to support resume
    let existing_keys = read_existing_points(&args.output_path);
    let all_points = generate_points(&args);
    let pending_points: Vec<(f64, f64)> = all_points
        .into_iter()
        .filter(|&(b, m)| !existing_keys.contains(&point_key(b, m)))
        .collect();

    println!(
        "Total grid points planned: {}. Already completed: {}. Remaining: {}.",
        existing_keys.len() + pending_points.len(),
        existing_keys.len(),
        pending_points.len()
    );

    if pending_points.is_empty() {
        println!("All planned points have already been computed! Nothing to do.");
        println!("You can run: python scripts/plot_uphill.py --input {}", args.output_path.display());
        return;
    }

    // Prepare CSV file and writer thread
    let is_new_file = !args.output_path.exists() || fs::metadata(&args.output_path).map(|m| m.len() == 0).unwrap_or(true);
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output_path)
        .expect("Failed to open output CSV file for appending");

    let (tx, rx) = mpsc::channel::<SimResult>();

    // Dedicated writer thread for non-blocking file I/O and immediate flushing
    let output_path_clone = args.output_path.clone();
    let writer_handle = std::thread::spawn(move || {
        let mut writer = BufWriter::new(file);
        if is_new_file {
            writeln!(
                writer,
                "beta,m_plus,l,therm_sweeps,meas_sweeps,sum_current,j_flux,j_per_sweep,final_mag,mean_mag,final_energy,zero_crossings,phase,is_metastable,is_uphill,elapsed_sec"
            ).expect("Failed to write CSV header");
            writer.flush().expect("Failed to flush CSV header");
        }

        while let Ok(res) = rx.recv() {
            writeln!(
                writer,
                "{:.6},{:.6},{},{},{},{},{:.8e},{:.8e},{:.6},{:.6},{:.6},{},{},{},{},{:.2}",
                res.beta,
                res.m_plus,
                res.l,
                res.therm_sweeps,
                res.meas_sweeps,
                res.sum_current,
                res.j_flux,
                res.j_per_sweep,
                res.final_mag,
                res.mean_mag,
                res.final_energy_per_spin,
                res.zero_crossings,
                res.phase,
                if res.is_metastable { 1 } else { 0 },
                if res.is_uphill { 1 } else { 0 },
                res.elapsed_sec
            ).expect("Failed to write simulation row");
            writer.flush().expect("Failed to flush CSV row");
        }
    });

    let total_pending = pending_points.len();
    let completed_counter = Arc::new(AtomicUsize::new(0));
    let uphill_counter = Arc::new(AtomicUsize::new(0));
    let metastable_counter = Arc::new(AtomicUsize::new(0));
    let global_start = Instant::now();

    // Process tasks in parallel across worker threads
    pending_points.par_iter().for_each(|&(beta, m_plus)| {
        let sim_config = SimConfig {
            l: args.l,
            beta,
            m_plus,
            therm_sweeps: args.therm_sweeps,
            meas_sweeps: args.meas_sweeps,
            replacement_mode: DemonReplacementMode::PerStep,
            bond_selection: BondSelection::Quenched,
            reservoir_type: ReservoirType::Quenched,
        };

        let result = run_simulation(&sim_config);

        let done = completed_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if result.is_uphill {
            uphill_counter.fetch_add(1, Ordering::Relaxed);
        }
        if result.is_metastable {
            metastable_counter.fetch_add(1, Ordering::Relaxed);
        }

        let status_tag = match result.phase.as_str() {
            "metastable" => "[\x1b[32mMETASTABLE UPHILL\x1b[0m]",
            "weakly_unstable" => "[\x1b[33mWEAKLY-UNSTABLE\x1b[0m]",
            "stable" => "[\x1b[34mSTABLE DOWNHILL\x1b[0m]",
            _ => "[\x1b[35mCHAOTIC\x1b[0m]",
        };

        let elapsed = global_start.elapsed().as_secs_f64();
        let eta_sec = if done > 0 {
            (elapsed / done as f64) * (total_pending - done) as f64
        } else {
            0.0
        };

        println!(
            "[{:>4}/{}] beta={:.4} m+={:.5} => J={:>+10.4e} <M>={:>+6.3} zc={} {} | took {:>5.2}s | ETA: {:>5.1}m",
            done,
            total_pending,
            result.beta,
            result.m_plus,
            result.j_flux,
            result.mean_mag,
            result.zero_crossings,
            status_tag,
            result.elapsed_sec,
            eta_sec / 60.0
        );

        tx.send(result).expect("Failed to send simulation result to writer thread");
    });

    // Drop sender to signal writer thread to finish
    drop(tx);
    writer_handle.join().expect("Writer thread panicked");

    let total_elapsed = global_start.elapsed().as_secs_f64();
    let uphill_total = uphill_counter.load(Ordering::Relaxed);
    let metastable_total = metastable_counter.load(Ordering::Relaxed);

    println!("============================================================");
    println!("Scan completed successfully in {:.1}s ({:.2}m)!", total_elapsed, total_elapsed / 60.0);
    println!("Points processed:    {}", total_pending);
    println!("Total uphill (J>0):  {}", uphill_total);
    println!("Metastable uphill:   {}", metastable_total);
    println!("Results saved to:    {}", output_path_clone.display());
    println!("To plot results, run:");
    println!("    python scripts/plot_uphill.py --input {}", output_path_clone.display());
    println!("============================================================");
}
