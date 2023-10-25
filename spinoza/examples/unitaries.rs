use spinoza::{
    core::State,
    gates::Gate,
    unitaries::{apply_unitary, Unitary},
};

fn main() {
    let n = 16;
    let state = State::new(n);
    let u = Unitary::from_single_qubit_gate(&state, Gate::U((1.0, 2.0, 3.0)), 0);

    let now = std::time::Instant::now();
    let _s = apply_unitary(&state, &u);
    let elapsed = now.elapsed().as_micros();
    println!("{elapsed} us elapsed");
}
