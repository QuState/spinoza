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

    let exp_vals = xyz_expectation_value('z', &state, &[target]);
    println!("expectation values: {:?}", exp_vals);

    let exp_val = qubit_expectation_value(&state, target);
    println!("{exp_val}");
}
