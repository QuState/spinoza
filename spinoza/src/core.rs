//! Abstractions for representing a Quantum State
use std::{collections::HashMap, fmt};

use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::OnceLock;

use crate::{
    config::Config,
    gates::{apply, c_apply, x_apply, y_apply, z_apply, Gate},
    math::{modulus, pow2f, Float, PI},
};

/// Reference to the Config for user passed config args
pub static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Clone)]
/// Representation of a Quantum State. Amplitudes are split between two vectors.
pub struct State {
    /// The real components of the state.
    pub reals: Vec<Float>,
    /// The imaginary components of the state.
    pub imags: Vec<Float>,
    /// The number of qubits represented by the state.
    pub n: u8,
}

impl State {
    /// Create a new State. The state will always be of size 2^{n},
    /// where n is the number of qubits. Note that n cannot be 0.
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        let mut reals = vec![0.0; 1 << n];
        let imags = vec![0.0; 1 << n];
        reals[0] = 1.0;
        Self {
            n: n as u8,
            reals,
            imags,
        }
    }

    /// Get the size of the state vector. Size of the state should always be
    /// 2^{n}, where n is the number of qubits.
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.imags.len()
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.reals
            .iter()
            .zip(self.imags.iter())
            .for_each(|(re, im)| {
                writeln!(f, "{re} + i{im}").unwrap();
            });
        Ok(())
    }
}

/// Reservoir for sampling
/// See <https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf>
pub struct Reservoir {
    entries: Vec<usize>,
    w_s: Float,
}

impl Reservoir {
    /// Create a new reservoir for sampling
    pub fn new(k: usize) -> Self {
        Self {
            entries: vec![0; k],
            w_s: 0.0,
        }
    }

    fn update(&mut self, e_i: usize, w_i: Float) {
        self.w_s += w_i;
        let delta = w_i / self.w_s;

        self.entries
            .par_iter_mut()
            .with_min_len(1 << 16)
            .for_each_init(thread_rng, |rng, e| {
                let epsilon_k: Float = rng.gen();
                if epsilon_k < delta {
                    *e = e_i;
                }
            });
    }

    fn weight(reals: &[Float], imags: &[Float], index: usize) -> Float {
        let (z_re, z_im) = (reals[index], imags[index]);
        modulus(z_re, z_im).powi(2)
    }

    /// Run the sampling based on the given State
    pub fn sampling(&mut self, reals: &[Float], imags: &[Float], num_tests: usize) {
        debug_assert_eq!(reals.len(), imags.len());
        let uniform_dist = Uniform::from(0..reals.len());
        let mut rng = thread_rng();

        self.w_s = 0.0;
        for _ in 0..num_tests {
            let outcome = uniform_dist.sample(&mut rng); // aka the index
            self.update(outcome, Self::weight(reals, imags, outcome));
        }
    }

    /// Create a histogram of the counts for each outcome
    pub fn get_outcome_count(&self) -> HashMap<usize, usize> {
        let mut samples = HashMap::new();
        for entry in self.entries.iter() {
            *samples.entry(*entry).or_insert(0) += 1;
        }
        samples
    }
}

/// Convenience function for running reservoir sampling
pub fn reservoir_sampling(state: &State, reservoir_size: usize, num_tests: usize) -> Reservoir {
    let mut reservoir = Reservoir::new(reservoir_size);
    reservoir.sampling(&state.reals, &state.imags, num_tests);
    reservoir
}

// pub fn g_proc(
//     state: &mut State,
//     range: std::ops::Range<usize>,
//     gate: &Gate,
//     marks: &[usize],
//     zeros: &std::collections::HashSet<usize>,
//     dist: usize,
// ) {
//     let n = marks.len();
//
//     let mut offset = 0;
//     for j in 0..n {
//         if !zeros.contains(&j) {
//             offset = offset + (1 << marks[j]);
//         }
//     }
//
//     for i in range {
//         let mut l1 = i;
//
//         for j in (0..n).rev() {
//             let k = marks[j] - j;
//             l1 = l1 + ((l1 >> k) << k);
//         }
//         l1 = l1 + offset;
//         apply(gate, state, l1 - dist, l1);
//     }
// }
//
// /// general transformation
// pub fn g_transform(
//     state: &mut [Amplitude],
//     controls: &[usize],
//     zeros: &HashSet<usize>,
//     target: usize,
//     gate: &impl Gate,
// ) {
//     let mut marks = [controls, &[target]].concat();
//     let num_pairs = state.len() >> marks.len();
//     marks.sort_unstable();
//
//     let data = SendPtr(state.as_mut_ptr());
//     let dist = 1 << target;
//
//     let range = Range {
//         start: 0,
//         end: num_pairs,
//     };
//
//     g_proc(data, range, gate, &marks, zeros, dist);
// }

/// Inverse Quantum Fourier transform
pub fn iqft(state: &mut State, targets: &[usize]) {
    for j in (0..targets.len()).rev() {
        apply(Gate::H, state, targets[j]);
        for k in (0..j).rev() {
            c_apply(Gate::P(-PI / pow2f(j - k)), state, targets[j], targets[k]);
        }
    }
}

// fn apply_bit_flip(prob: Float, target: usize) {
//     todo!()
// }

/// Compute the expectation value of a qubit measurement.
pub fn qubit_expectation_value(state: &State, target: usize) -> Float {
    let chunk_size = 1 << (target + 1);
    let dist = 1 << target;

    let prob0 = state
        .reals
        .par_chunks_exact(chunk_size)
        .zip_eq(state.imags.par_chunks_exact(chunk_size))
        .map(|(reals_chunk, imags_chunk)| {
            reals_chunk
                .par_iter()
                .take(dist)
                .zip_eq(imags_chunk.par_iter().take(dist))
                .with_min_len(1 << 16)
                .map(|(re_s0, im_s0)| re_s0.powi(2) + im_s0.powi(2))
                .sum::<Float>()
        })
        .sum::<Float>();

    // p0 - p1 == p0 - (1 - p0) == p0 - 1 + p0 == 2p0 - 1
    2.0 * prob0 - 1.0
}

/// Compute the expectation value of certain observables (either X, Y, or Z) in the given state.
pub fn xyz_expectation_value(observable: char, state: &State, targets: &[usize]) -> Vec<Float> {
    if !"xyz".contains(observable) {
        panic!("observable {observable} not supported");
    }

    let mut working_state = state.clone();
    let mut values = Vec::with_capacity(targets.len());

    for target in targets.iter() {
        if observable == 'z' {
            z_apply(&mut working_state, *target);
        } else if observable == 'y' {
            y_apply(&mut working_state, *target);
        } else {
            x_apply(&mut working_state, *target);
        }

        // v = O * psi
        // <psi | O | psi>

        // <psi | v >
        // (a + ib) * (c + id) = a * c + ibc + iad - bd = ac - bd + i(bc + ad)
        let k_re = (
            &state.reals,
            &state.imags,
            &working_state.reals,
            &working_state.imags,
        )
            .into_par_iter()
            .map(|(s_re, s_im, v_re, v_im)| {
                let a = *s_re;
                let b = *s_im;
                let c = v_re;
                let d = v_im;
                a * c + b * d
            })
            .sum();

        values.push(k_re);
        working_state = state.clone();
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::assert_float_closeness;

    #[test]
    fn encoded_integers() {
        const N: usize = 3;

        let state = State::new(N);
        let reservoir = reservoir_sampling(&state, state.len(), state.len() * 10_000);
        let histogram = reservoir.get_outcome_count();
        let count = *histogram.get(&0).unwrap();
        assert_eq!(count, state.len());

        for i in 1..(1 << N) {
            let mut state = State::new(N);
            state.reals[0] = 0.0;
            state.reals[i] = 1.0;

            let reservoir = reservoir_sampling(&state, state.len(), state.len() * 10_000);
            let histogram = reservoir.get_outcome_count();
            let count = *histogram.get(&i).unwrap();
            assert_eq!(count, state.len());
        }
    }

    #[test]
    fn xyz_exp_val() {
        let mut state = State::new(1);

        apply(Gate::RX(0.54), &mut state, 0);
        apply(Gate::RY(0.12), &mut state, 0);
        let exp_vals = xyz_expectation_value('z', &state, &[0]);
        assert_float_closeness(exp_vals[0], 0.8515405859048367, 0.0001);
    }

    #[test]
    #[should_panic]
    fn xyz_exp_val_bad_observable() {
        let mut state = State::new(1);
        apply(Gate::RX(0.54), &mut state, 0);
        apply(Gate::RY(0.12), &mut state, 0);
        xyz_expectation_value('a', &state, &[0]);
    }

    #[test]
    fn xyz_exp_val_x_as_observable() {
        let mut state = State::new(1);
        apply(Gate::RX(0.54), &mut state, 0);
        apply(Gate::RY(0.12), &mut state, 0);
        xyz_expectation_value('x', &state, &[0]);
    }

    #[test]
    fn xyz_exp_val_y_as_observable() {
        let mut state = State::new(1);
        apply(Gate::RX(0.54), &mut state, 0);
        apply(Gate::RY(0.12), &mut state, 0);
        xyz_expectation_value('y', &state, &[0]);
    }

    #[test]
    fn qubit_exp_val() {
        let mut state = State::new(1);
        let target = 0;

        apply(Gate::RX(0.54), &mut state, target);
        apply(Gate::RY(0.12), &mut state, target);
        let exp_vals = xyz_expectation_value('z', &state, &[target]);
        let qubit_exp_val = qubit_expectation_value(&state, target);

        assert_float_closeness(qubit_exp_val, exp_vals[0], 0.0001);
    }
}
