use crate::{core::State, math::modulus};
use rand_distr::{Binomial, Distribution};

/// Single qubit measurement
pub fn measure_qubit(state: &mut State, target: usize, v: Option<u64>) {
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

    let bin = Binomial::new(1, prob1).unwrap();

    let val = if let Some(_v) = v {
        assert!(_v == 0 || _v == 1);
        _v
    } else {
        bin.sample(&mut rand::thread_rng())
    };

    if val == 0 {
        for i in 0..num_pairs {
            let s0 = i + ((i >> target) << target);
            let s1 = s0 + distance;

            state.reals[s0] = 0.0;
            state.imags[s0] = 0.0;
            state.reals[s1] /= prob1.sqrt();
            state.imags[s1] /= prob1.sqrt();
        }
    } else {
        for i in 0..num_pairs {
            let s0 = i + ((i >> target) << target);
            let s1 = s0 + distance;

            state.reals[s0] /= prob0.sqrt();
            state.imags[s0] /= prob0.sqrt();
            state.reals[s1] = 0.0;
            state.imags[s1] = 0.0;
        }
    }
}
