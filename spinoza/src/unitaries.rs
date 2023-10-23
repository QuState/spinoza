//! Functionality for applying large 2^n * 2^n matrices to the state
//! Ideally, this should be a last resort
use crate::{
    core::State,
    math::{Float, SQRT_ONE_HALF},
};

/// A representation of a Unitary Matrix
pub struct Unitary {
    reals: Vec<Float>,
    imags: Vec<Float>,
    /// The number of rows in the matrix
    height: usize,
    /// The number of columns in the matrix
    width: usize,
}

/// Applies a unitary matrix to the Quantum State Vector
pub fn apply_unitary(state: &mut State, unitary: &Unitary) -> State {
    assert!(state.len() == unitary.width && state.len() == unitary.height);
    let chunk_size = unitary.width;

    let mut s = state.clone();

    unitary
        .reals
        .chunks_exact(chunk_size)
        .zip(unitary.imags.chunks_exact(chunk_size))
        .enumerate()
        .for_each(|(i, (row_reals, row_imags))| {
            let mut dot_prod_re = 0.0;
            let mut dot_prod_im = 0.0;
            let c = state.reals[i];
            let d = state.imags[i];
            row_reals.iter().zip(row_imags.iter()).for_each(|(a, b)| {
                dot_prod_re += *a * c - *b * d;
                dot_prod_im += *a * d + *b * c;
            });
            s.reals[i] = dot_prod_re;
            s.imags[i] = dot_prod_im;
        });
    s
}

/// The gate H 2 = H ⊗ H {\displaystyle H_{2}=H\otimes H} is the Hadamard gate (H) applied in
/// parallel on 2 qubits
fn generate_h2() -> Unitary {
    let mut reals = vec![
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0,
    ];
    reals.iter_mut().for_each(|a| *a *= SQRT_ONE_HALF);

    let imags = vec![0.0; reals.len()];

    Unitary {
        reals,
        imags,
        height: 4,
        width: 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{apply, Gate};

    #[test]
    fn test_h2() {
        let N = 2;
        let mut state = State::new(N);
        let u = generate_h2();

        let s = apply_unitary(&mut state, &u);
        println!("old state:\n{state}");
        println!("new state:\n{s}");

        let mut s1 = State::new(N);
        println!("old state:\n{s1}");
        apply(Gate::H, &mut s1, 0);
        println!("new s1:\n{s}");
    }
}
