use clap::Parser;
use spinoza::{
    config::{Config, QSArgs},
    core::{iqft, State, CONFIG},
    gates::{apply, Gate},
    math::{pow2f, PI},
    utils::{pretty_print_int, to_table},
};

fn value_encoding(n: usize, show_results: bool) {
    let v = 2.4;
    let now = std::time::Instant::now();
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::H, &mut state, i);
    }
    for i in 0..n {
        apply(Gate::P(2.0 * PI / (pow2f(i + 1)) * v), &mut state, i);
    }
    let targets: Vec<usize> = (0..n).rev().collect();

    iqft(&mut state, &targets);

    let elapsed = now.elapsed().as_micros();
    println!("{}", pretty_print_int(elapsed));
    if show_results {
        println!("{}", to_table(&state));
    }
}

pub fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    value_encoding(config.qubits.into(), config.print);
}
