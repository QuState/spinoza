//! Functionality for measurement
use crate::{
    core::State,
    gates::{apply, Gate},
    math::modulus,
};
use rand_distr::{Binomial, Distribution};

/// Single qubit measurement
pub fn measure_qubit(state: &mut State, target: usize, reset: bool, v: Option<u64>) -> u64 {
    let mut prob0 = 0.0;
    let mut prob1 = 0.0;
    let num_pairs = state.len() >> 1;
    let distance = 1 << target;

    for i in 0..num_pairs {
        let s0 = i + ((i >> target) << target);
        let s1 = s0 + distance;

        prob0 += modulus(state.reals[s0], state.imags[s0]).powi(2);
        prob1 += modulus(state.reals[s1], state.imags[s1]).powi(2);
    }

    let val = if let Some(_v) = v {
        assert!(_v == 0 || _v == 1);
        _v
    } else {
        let bin = Binomial::new(1, prob1).unwrap();
        bin.sample(&mut rand::thread_rng())
    };

    if val == 0 {
        for i in 0..num_pairs {
            let s0 = i + ((i >> target) << target);
            let s1 = s0 + distance;

            state.reals[s0] /= prob0.sqrt();
            state.imags[s0] /= prob0.sqrt();
            state.reals[s1] = 0.0;
            state.imags[s1] = 0.0;
        }
    } else {
        for i in 0..num_pairs {
            let s0 = i + ((i >> target) << target);
            let s1 = s0 + distance;

            state.reals[s0] = 0.0;
            state.imags[s0] = 0.0;
            state.reals[s1] /= prob1.sqrt();
            state.imags[s1] /= prob1.sqrt();
        }

        if reset {
            apply(Gate::X, state, target);
        }
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_float_closeness, gen_random_state};

    #[test]
    fn test_measure_qubit() {
        let mut state = gen_random_state(3);
        println!("{state}");
        let sum = state
            .reals
            .iter()
            .zip(state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();
        assert_float_closeness(sum, 1.0, 0.001);

        measure_qubit(&mut state, 0, true, Some(0));
        println!("{state}");
        let sum = state
            .reals
            .iter()
            .zip(state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();
        assert_float_closeness(sum, 1.0, 0.001);

        measure_qubit(&mut state, 1, true, Some(0));
        println!("{state}");
        let sum = state
            .reals
            .iter()
            .zip(state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();
        assert_float_closeness(sum, 1.0, 0.001);

        measure_qubit(&mut state, 2, true, Some(1));
        println!("{state}");
        let sum = state
            .reals
            .iter()
            .zip(state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();

        assert_float_closeness(sum, 1.0, 0.001);
    }
}
