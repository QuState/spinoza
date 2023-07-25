//! Abstractions for representing a Quantum State
use crate::{
    config::Config,
    gates::{apply, c_apply, Gate},
    math::{modulus, pow2f, Float, PI},
};
use once_cell::sync::OnceCell;
use rand::prelude::*;
use std::collections::HashMap;

/// Reference to the Config for user passed config args
pub static CONFIG: OnceCell<Config> = OnceCell::new();

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

/// Reservoir for sampling
/// See https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf
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

    fn update(&mut self, e_i: usize, w_i: Float, rng: &mut ThreadRng) {
        self.w_s += w_i;
        let delta = w_i / self.w_s;

        self.entries.iter_mut().for_each(|e| {
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
    pub fn sampling(&mut self, reals: &[Float], imags: &[Float], m: usize) {
        let mut rng = thread_rng();
        let mut outcomes = (0..reals.len()).cycle();

        for _ in 0..m {
            let outcome = outcomes.next().unwrap(); // aka the index
            self.update(outcome, Self::weight(reals, imags, outcome), &mut rng);
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
pub fn reservoir_sampling(state: &State, k: usize) -> Reservoir {
    let m = 1 << 10;
    let mut reservoir = Reservoir::new(k);
    reservoir.sampling(&state.reals, &state.imags, m);
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

/// Swap using controlled X gates
pub fn swap(state: &mut State, first: usize, second: usize) {
    c_apply(Gate::X, state, first, second);
    c_apply(Gate::X, state, second, first);
    c_apply(Gate::X, state, first, second);
}

/// Inverse Quantum Fourier transform
pub fn iqft(state: &mut State, targets: &[usize]) {
    for j in (0..targets.len()).rev() {
        apply(Gate::H, state, targets[j]);
        for k in (0..j).rev() {
            c_apply(Gate::P(-PI / pow2f(j - k)), state, targets[j], targets[k]);
        }
    }
}
