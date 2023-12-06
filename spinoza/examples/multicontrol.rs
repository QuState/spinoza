use clap::Parser;
use spinoza::{
    config::{Config, QSArgs},
    core::{State, CONFIG},
    gates::{mc_apply, Gate},
    utils::to_table,
};

fn mcx(_n: usize, show_results: bool) {
    let mut state = State::new(4);

    mc_apply(Gate::X, &mut state, &[1, 2], None, 0);

    if show_results {
        println!("{}", to_table(&state));
    }
}

fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    mcx(config.qubits.into(), config.print);
}
