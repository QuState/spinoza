use clap::Parser;
use spinoza::core::qubit_expectation_value;
use spinoza::{
    config::{Config, QSArgs},
    core::{xyz_expectation_value, State, CONFIG},
    gates::{apply, Gate},
};

fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();

    let n = config.qubits.into();
    let mut state = State::new(n);
    let target = 0;

    apply(Gate::RX(0.54), &mut state, target);
    apply(Gate::RY(0.12), &mut state, target);

    let targets = (0..n).collect::<Vec<usize>>();
    let exp_vals = xyz_expectation_value('z', &state, &targets);
    println!("expectation values: {:?}", exp_vals);

    let exp_vals: Vec<_> = (0..n).map(|t| qubit_expectation_value(&state, t)).collect();
    println!(
        "expectation values using `qubit_expectation_value`: {:?}",
        exp_vals
    );
}
