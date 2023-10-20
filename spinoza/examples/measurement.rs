use clap::Parser;
use spinoza::{
    config::{Config, QSArgs},
    core::CONFIG,
    measurement::measure_qubit,
    utils::gen_random_state,
};

fn measure_qubits(n: usize) {
    let mut state = gen_random_state(n);

    state
        .reals
        .iter()
        .zip(state.imags.iter())
        .for_each(|(re, im)| {
            println!("{re},{im}");
        });

    println!("----------------------------------------");
    println!("{state}");

    measure_qubit(&mut state, 0, true, Some(0));
    println!("{state}");

    measure_qubit(&mut state, 1, true, Some(0));
    println!("{state}");

    measure_qubit(&mut state, 2, true, Some(1));
    println!("{state}");
}

fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    measure_qubits(config.qubits.into());
}
