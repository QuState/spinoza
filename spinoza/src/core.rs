use crate::{
    config::Config,
    gates::{apply, c_apply, Gate},
    math::{modulus, pow2f, Float, PI},
};
use once_cell::sync::OnceCell;
use rand::prelude::*;
use std::collections::HashMap;

pub static CONFIG: OnceCell<Config> = OnceCell::new();

#[derive(Clone)]
pub struct State {
    pub reals: Vec<Float>,
    pub imags: Vec<Float>,
    pub n: u8,
}

impl State {
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

    #[inline]
    pub fn len(&self) -> usize {
        self.imags.len()
    }
}

pub struct Reservoir {
    pub entries: Vec<usize>,
    w_s: Float,
}

impl Reservoir {
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

    pub fn sampling(&mut self, reals: &[Float], imags: &[Float], m: usize) {
        let mut rng = thread_rng();
        let mut outcomes = (0..reals.len()).cycle();

        for _ in 0..m {
            let outcome = outcomes.next().unwrap(); // aka the index
            self.update(outcome, Self::weight(reals, imags, outcome), &mut rng);
        }
    }

    pub fn get_outcome_count(&self) -> HashMap<usize, usize> {
        let mut samples = HashMap::new();
        for entry in self.entries.iter() {
            *samples.entry(*entry).or_insert(0) += 1;
        }
        samples
    }
}

pub fn reservoir_sampling(state: &State, k: usize) -> Reservoir {
    let m = 1 << 10;
    let mut reservoir = Reservoir::new(k);
    reservoir.sampling(&state.reals, &state.imags, m);
    reservoir
}

// TODO(Saveliy Yusufov): get multiple reservoirs working to parallelize measurement
// pub fn multi_reservoir_sampling(state: &State, k: usize) -> Reservoir {
//     let cpus = num_cpus::get();
//     let m = 1 << 10;
//
//     let mut reservoirs: Vec<Reservoir> = (0..cpus).map(|_| Reservoir::new(k)).collect();
//
//     let mut i = 0;
//     for (chunk_reals, chunk_imags) in state.reals.chunks(cpus).zip(state.imags.chunks(cpus)) {
//         reservoirs[i].sampling(chunk_reals, chunk_imags, m);
//         i += 1;
//     }
//
//     reservoirs
//         .into_iter()
//         .fold(Reservoir::new(k), |r0, r1| merge_reservoirs(&r0, &r1, k))
// }
//
// fn merge_reservoirs(r0: &Reservoir, r1: &Reservoir, k: usize) -> Reservoir {
//     let mut rng = thread_rng();
//     let mut res = Reservoir::new(k);
//
//     let (w_0, w_1) = (r0.w_s, r1.w_s);
//     let prob = w_0 / (w_0 + w_1);
//
//     for i in 0..k {
//         let epsilon_i: Float = rng.gen();
//         if epsilon_i <= prob {
//             res.entries[i] = r0.entries[i];
//         } else {
//             res.entries[i] = r1.entries[i];
//         }
//     }
//     res
// }

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

pub fn swap(state: &mut State, first: usize, second: usize) {
    c_apply(Gate::X, state, first, second);
    c_apply(Gate::X, state, second, first);
    c_apply(Gate::X, state, first, second);
}

pub fn iqft(state: &mut State, targets: &[usize]) {
    for j in (0..targets.len()).rev() {
        apply(Gate::H, state, targets[j]);
        for k in (0..j).rev() {
            c_apply(Gate::P(-PI / pow2f(j - k)), state, targets[j], targets[k]);
        }
    }
}
