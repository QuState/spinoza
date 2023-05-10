use clap::Parser;
use spinoza::{
    config::{Config, QSArgs},
    core::{State, CONFIG},
    gates::{apply, Gate},
    utils::{pretty_print_int, to_table},
};

fn pauli_functional(n: usize, show_results: bool) {
    let now = std::time::Instant::now();
    let mut state = State::new(n);

    apply(Gate::X, &mut state, 0);

    for _ in 1..(1 << 30) {
        apply(Gate::H, &mut state, 0);
        apply(Gate::X, &mut state, 0);
        apply(Gate::Z, &mut state, 0);
    }

    let elapsed = now.elapsed().as_micros();
    println!("{}", pretty_print_int(elapsed));
    if show_results {
        to_table(&state);
    }
}

pub fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();

    pauli_functional(config.qubits.into(), config.print);
}
