//! Abstractions for quantum logic gates
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::unitaries::Unitary;
use crate::{
    config::Config,
    consts::{H, X, Y, Z},
    core::State,
    math::{Amplitude, Float, SQRT_ONE_HALF},
};

const LOW_QUBIT_THRESHOLD: u8 = 15;

// https://github.com/rayon-rs/rayon/blob/master/src/lib.rs
struct SendPtr<T>(*mut T);

// SAFETY: !Send for raw pointers is not for safety, just as a lint
unsafe impl<T: Send> Send for SendPtr<T> {}

// SAFETY: !Sync for raw pointers is not for safety, just as a lint
unsafe impl<T: Send> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    // Helper to avoid disjoint captures of `send_ptr.0`
    fn get(self) -> *mut T {
        self.0
    }
}

// Implement Clone without the T: Clone bound from the derive
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

// Implement Copy without the T: Copy bound from the derive
impl<T> Copy for SendPtr<T> {}

/// Quantum Logic Gates
/// See <https://en.wikipedia.org/wiki/Quantum_logic_gate> for more info
#[derive(Clone)]
pub enum Gate {
    /// Hadamard gate. See <https://en.wikipedia.org/wiki/Quantum_logic_gate#Hadamard_gate>
    H,
    /// Measurement 'gate'
    M,
    /// The Pauli-X gate is the quantum equivalent of the NOT gate for classical computers with
    /// respect to the standard basis |0>, |1>. See
    /// <https://en.wikipedia.org/wiki/Quantum_logic_gate#Pauli_gates_(X,Y,Z)>
    X,
    /// See <https://en.wikipedia.org/wiki/Quantum_logic_gate#Pauli_gates_(X,Y,Z)>
    Y,
    /// See <https://en.wikipedia.org/wiki/Quantum_logic_gate#Pauli_gates_(X,Y,Z)>
    Z,
    /// Phase shift gate. See <https://en.wikipedia.org/wiki/Quantum_logic_gate#Phase_shift_gates>
    P(Float),
    /// Rx gate for rotation about the x-axis. See <https://en.wikipedia.org/wiki/List_of_quantum_logic_gates#Rotation_operator_gates>
    RX(Float),
    /// Ry gate for rotation about the y-axis. See <https://en.wikipedia.org/wiki/List_of_quantum_logic_gates#Rotation_operator_gates>
    RY(Float),
    /// Rz gate for rotation about the z-axis. See <https://en.wikipedia.org/wiki/List_of_quantum_logic_gates#Rotation_operator_gates>
    RZ(Float),
    /// Swap gate swaps two qubits. See <https://en.wikipedia.org/wiki/Quantum_logic_gate#Swap_gate>
    SWAP(usize, usize),
    /// General single qubit rotation. See <https://en.wikipedia.org/wiki/List_of_quantum_logic_gates#Other_named_gates>
    U(Float, Float, Float),
    /// A Unitary matrix. See <https://mathworld.wolfram.com/UnitaryMatrix.html>
    Unitary(Unitary),
    /// A gate to simulate a bit flip based on the provided probability.
    BitFlipNoise(Float),
}

impl Gate {
    /// Return the inverted gate
    pub fn inverse(self) -> Self {
        match self {
            Self::H | Self::X | Self::Y | Self::Z | Self::SWAP(_, _) => self,
            Self::P(theta) => Self::P(-theta),
            Self::RX(theta) => Self::RX(-theta),
            Self::RY(theta) => Self::RY(-theta),
            Self::RZ(theta) => Self::RZ(-theta),
            Self::U(theta, phi, lambda) => Self::U(-theta, -lambda, -phi),
            Self::M | Self::BitFlipNoise(_) => unimplemented!(),
            Self::Unitary(mut unitary) => {
                unitary.conj_t();
                Self::Unitary(unitary)
            }
        }
    }

    /// Return the 2 x 2 matrix representation of the gate
    pub fn to_matrix(&self) -> [Amplitude; 4] {
        match self {
            Self::H => H,
            Self::X => X,
            Self::Y => Y,
            Self::Z => Z,
            Self::P(theta) => [
                Amplitude { re: 1.0, im: 0.0 },
                Amplitude { re: 0.0, im: 0.0 },
                Amplitude { re: 0.0, im: 0.0 },
                Amplitude {
                    re: theta.cos(),
                    im: theta.sin(),
                },
            ],
            Self::RX(theta) => {
                let theta = theta / 2.0;
                [
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: 0.0,
                        im: -theta.sin(),
                    },
                    Amplitude {
                        re: 0.0,
                        im: -theta.sin(),
                    },
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                ]
            }
            Self::RY(theta) => {
                let theta = theta / 2.0;
                [
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: -theta.sin(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: theta.sin(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                ]
            }
            Self::RZ(theta) => {
                let theta = theta / 2.0;
                [
                    Amplitude {
                        re: theta.cos(),
                        im: -theta.sin(),
                    },
                    Amplitude { re: 0.0, im: 0.0 },
                    Amplitude { re: 0.0, im: 0.0 },
                    Amplitude {
                        re: theta.cos(),
                        im: theta.sin(),
                    },
                ]
            }
            Self::U(theta, phi, lambda) => {
                let theta = theta / 2.0;
                [
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: -lambda.cos() * theta.sin(),
                        im: -lambda.sin() * theta.sin(),
                    },
                    Amplitude {
                        re: phi.cos() * theta.sin(),
                        im: phi.sin() * theta.sin(),
                    },
                    Amplitude {
                        re: (phi + lambda).cos() * theta.cos(),
                        im: (phi + lambda).sin() * theta.cos(),
                    },
                ]
            }
            _ => unimplemented!(),
        }
    }
}

/// Apply a transformation to a single target qubit, with no control(s).
///
/// # Examples
/// ```
/// use spinoza::{gates::{apply, Gate}, core::State};
///
/// let n = 3;
/// let mut state = State::new(n);
///
/// for t in 0..n {
///     apply(Gate::H, &mut state, t);
/// }
/// ```
#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
pub fn apply(gate: Gate, state: &mut State, target: usize) {
    match gate {
        Gate::H => h_apply(state, target),
        Gate::X => x_apply(state, target),
        Gate::Y => y_apply(state, target),
        Gate::Z => z_apply(state, target),
        Gate::P(theta) => p_apply(state, target, theta),
        Gate::RX(theta) => rx_apply(state, target, theta),
        Gate::RY(theta) => ry_apply(state, target, theta),
        Gate::RZ(theta) => rz_apply(state, target, theta),
        Gate::SWAP(t0, t1) => swap_apply(state, t0, t1),
        Gate::U(theta, phi, lambda) => u_apply(state, target, theta, phi, lambda),
        Gate::BitFlipNoise(prob) => {
            bit_flip_noise_apply(state, prob, target);
        }
        _ => unimplemented!(),
    }
}

/// Apply a transformation to a single target qubit, with a single control.
///
/// # Examples
/// ```
/// use spinoza::{gates::{c_apply, Gate}, core::State};
///
/// let n = 3;
/// let mut state = State::new(n);
///
/// let pairs: Vec<(usize, usize)> = (0..n).map(|i| (i, (i +1) % n)).collect();
/// for (control, target) in pairs.iter() {
///     c_apply(Gate::H, &mut state, *control, *target);
/// }
/// ```
#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
pub fn c_apply(gate: Gate, state: &mut State, control: usize, target: usize) {
    match gate {
        Gate::H => h_c_apply(state, control, target),
        Gate::X => x_c_apply(state, control, target),
        Gate::Y => y_c_apply(state, control, target),
        Gate::P(theta) => p_c_apply(state, control, target, theta),
        Gate::RX(theta) => rx_c_apply(state, control, target, theta),
        Gate::RY(theta) => ry_c_apply(state, control, target, theta),
        Gate::RZ(theta) => rz_c_apply(state, control, target, theta),
        Gate::U(theta, phi, lambda) => u_c_apply(state, theta, phi, lambda, control, target),
        _ => todo!(),
    }
}

/// Two Controls, Single Target
pub fn cc_apply(gate: Gate, state: &mut State, control0: usize, control1: usize, target: usize) {
    match gate {
        Gate::X => x_cc_apply(state, control0, control1, target),
        _ => todo!(),
    }
}

/// Apply a transformation to a single target qubit, with multiple controls.
///
/// # Examples
/// ```
/// use spinoza::{gates::{mc_apply, Gate}, core::State};
///
/// let n = 3;
/// let mut state = State::new(n);
///
/// mc_apply(Gate::P(3.14), &mut state, &[0, 1], None, 2);
/// ```
pub fn mc_apply(
    gate: Gate,
    state: &mut State,
    controls: &[usize],
    zeros: Option<HashSet<usize>>,
    target: usize,
) {
    debug_assert!(usize::from(state.n) > controls.len());
    let mut mask: u64 = 0;

    if let Some(z) = zeros {
        for control in controls.iter() {
            if z.contains(control) {
                continue;
            }
            mask |= 1 << control
        }
    } else {
        for control in controls.iter() {
            mask |= 1 << control
        }
    }

    match gate {
        Gate::X => x_mc_apply(state, mask, target),
        Gate::P(theta) => p_mc_apply(state, mask, target, theta),
        Gate::RX(theta) => rx_mc_apply(state, mask, target, theta),
        Gate::RY(theta) => ry_mc_apply(state, mask, target, theta),
        _ => todo!(),
    }
}

fn x_apply_target_0(state_re: SendPtr<Float>, state_im: SendPtr<Float>, s0: usize) {
    unsafe {
        std::ptr::swap(state_re.get().add(s0), state_re.get().add(s0 + 1));
        std::ptr::swap(state_im.get().add(s0), state_im.get().add(s0 + 1));
    }
}

fn x_apply_target(state_re: SendPtr<Float>, state_im: SendPtr<Float>, s0: usize, s1: usize) {
    unsafe {
        std::ptr::swap(state_re.get().add(s0), state_re.get().add(s1));
        std::ptr::swap(state_im.get().add(s0), state_im.get().add(s1));
    }
}

fn x_proc_chunk(state_re: SendPtr<Float>, state_im: SendPtr<Float>, chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let s0 = base + i;
        let s1 = s0 + dist;
        x_apply_target(state_re, state_im, s0, s1)
    }
}

pub(crate) fn x_apply(state: &mut State, target: usize) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    if target == 0 {
        if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
            (0..state.len()).step_by(2).for_each(|s0| {
                x_apply_target_0(state_re, state_im, s0);
            });
        } else {
            (0..state.len()).into_par_iter().step_by(2).for_each(|s0| {
                x_apply_target_0(state_re, state_im, s0);
            });
        }
    } else {
        let end = state.len() >> 1;
        let chunks = end >> target;

        if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
            (0..chunks).for_each(|chunk| {
                x_proc_chunk(state_re, state_im, chunk, target);
            });
        } else {
            (0..chunks).into_par_iter().for_each(|chunk| {
                x_proc_chunk(state_re, state_im, chunk, target);
            });
        }
    }
}

fn x_c_apply(state: &mut State, control: usize, target: usize) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    let end = state.len() >> 2;
    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..end).for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;
            x_apply_target(state_re, state_im, s0, s1);
        });
    } else {
        (0..end).into_par_iter().for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;
            x_apply_target(state_re, state_im, s0, s1);
        });
    }
}

fn x_cc_apply(state: &mut State, control0: usize, control1: usize, target: usize) {
    let mut i = 0;
    let dist = 1 << target;

    while i < state.len() {
        let c0c1_set = (((i >> control0) & 1) != 0) && (((i >> control1) & 1) != 0);
        if c0c1_set {
            let s0 = i;
            let s1 = i + dist;
            unsafe {
                let temp0 = *state.reals.get_unchecked(s0);
                *state.reals.get_unchecked_mut(s0) = *state.reals.get_unchecked(s1);
                *state.reals.get_unchecked_mut(s1) = temp0;

                let temp1 = *state.imags.get_unchecked(s0);
                *state.imags.get_unchecked_mut(s0) = *state.imags.get_unchecked(s1);
                *state.imags.get_unchecked_mut(s1) = temp1;
            };
            i += dist;
        }
        i += 1;
    }
}

fn x_mc_apply(state: &mut State, mask: u64, target: usize) {
    let mut i = 0;
    let dist = 1 << target;

    while i < state.len() {
        if (i as u64 & mask) == mask {
            let s0 = i;
            let s1 = i + dist;
            unsafe {
                let temp0 = *state.reals.get_unchecked(s0);
                *state.reals.get_unchecked_mut(s0) = *state.reals.get_unchecked(s1);
                *state.reals.get_unchecked_mut(s1) = temp0;

                let temp1 = *state.imags.get_unchecked(s0);
                *state.imags.get_unchecked_mut(s0) = *state.imags.get_unchecked(s1);
                *state.imags.get_unchecked_mut(s1) = temp1;
            };
            i += dist;
        }
        i += 1;
    }
}

fn y_proc_chunk(state_re: SendPtr<Float>, state_im: SendPtr<Float>, chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let s0 = base + i;
        let s1 = s0 + dist;

        unsafe {
            std::ptr::swap(state_re.get().add(s0), state_re.get().add(s1));
            std::ptr::swap(state_im.get().add(s0), state_im.get().add(s1));

            std::ptr::swap(state_re.get().add(s0), state_im.get().add(s0));
            std::ptr::swap(state_re.get().add(s1), state_im.get().add(s1));

            *state_im.get().add(s0) = -(*state_im.get().add(s0));
            *state_re.get().add(s1) = -(*state_re.get().add(s1));
        }
    }
}

pub(crate) fn y_apply(state: &mut State, target: usize) {
    let end = state.len() >> 1;
    let chunks = end >> target;

    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..chunks).for_each(|chunk| {
            y_proc_chunk(state_re, state_im, chunk, target);
        });
    } else {
        (0..chunks).into_par_iter().for_each(|chunk| {
            y_proc_chunk(state_re, state_im, chunk, target);
        });
    }
}

fn y_c_apply_to_range(state: &mut State, control: usize, target: usize) {
    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));
    let end = state.len() >> 2;

    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        for i in 0..end {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;

            unsafe {
                std::ptr::swap(state_re.get().add(s0), state_re.get().add(s1));
                std::ptr::swap(state_im.get().add(s0), state_im.get().add(s1));

                std::ptr::swap(state_re.get().add(s0), state_im.get().add(s0));
                std::ptr::swap(state_re.get().add(s1), state_im.get().add(s1));

                *state_im.get().add(s0) = -(*state_im.get().add(s0));
                *state_re.get().add(s1) = -(*state_re.get().add(s1));
            }
        }
    } else {
        (0..end).into_par_iter().for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;

            unsafe {
                std::ptr::swap(state_re.get().add(s0), state_re.get().add(s1));
                std::ptr::swap(state_im.get().add(s0), state_im.get().add(s1));

                std::ptr::swap(state_re.get().add(s0), state_im.get().add(s0));
                std::ptr::swap(state_re.get().add(s1), state_im.get().add(s1));

                *state_im.get().add(s0) = -(*state_im.get().add(s0));
                *state_re.get().add(s1) = -(*state_re.get().add(s1));
            }
        });
    }
}

fn y_c_apply(state: &mut State, control: usize, target: usize) {
    y_c_apply_to_range(state, control, target);
}

/// Apply H gate via strategy 2
fn h_apply_strat2(state: &mut State, chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let s0 = base + i;
        let s1 = s0 + dist;

        let (a, b, c, d) = unsafe {
            let a = *state.reals.get_unchecked(s0);
            let b = *state.imags.get_unchecked(s0);
            let c = *state.reals.get_unchecked(s1);
            let d = *state.imags.get_unchecked(s1);
            (a, b, c, d)
        };

        let a1 = SQRT_ONE_HALF * a;
        let b1 = SQRT_ONE_HALF * b;
        let c1 = SQRT_ONE_HALF * c;
        let d1 = SQRT_ONE_HALF * d;

        unsafe {
            *state.reals.get_unchecked_mut(s0) = a1 + c1;
            *state.imags.get_unchecked_mut(s0) = b1 + d1;
            *state.reals.get_unchecked_mut(s1) = a1 - c1;
            *state.imags.get_unchecked_mut(s1) = b1 - d1;
        }
    }
}

fn h_apply_strat2_par(
    state_re: SendPtr<Float>,
    state_im: SendPtr<Float>,
    chunk: usize,
    target: usize,
) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;

    for i in 0..dist {
        let s0 = base + i;
        let s1 = s0 + dist;

        let (a, b, c, d) = unsafe {
            let a = *state_re.get().add(s0);
            let b = *state_im.get().add(s0);
            let c = *state_re.get().add(s1);
            let d = *state_im.get().add(s1);
            (a, b, c, d)
        };

        let a1 = SQRT_ONE_HALF * a;
        let b1 = SQRT_ONE_HALF * b;
        let c1 = SQRT_ONE_HALF * c;
        let d1 = SQRT_ONE_HALF * d;

        unsafe {
            *state_re.get().add(s0) = a1 + c1;
            *state_im.get().add(s0) = b1 + c1;
            *state_re.get().add(s1) = a1 - c1;
            *state_im.get().add(s1) = b1 - d1;
        }
    }
}

fn h_apply(state: &mut State, target: usize) {
    let end = state.len() >> 1;
    let chunks = end >> target;

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..chunks).for_each(|c| {
            h_apply_strat2(state, c, target);
        });
    } else {
        let state_re = SendPtr(state.reals.as_mut_ptr());
        let state_im = SendPtr(state.imags.as_mut_ptr());

        (0..chunks).into_par_iter().for_each(|c| {
            h_apply_strat2_par(state_re, state_im, c, target);
        });
    }
}

fn h_c_apply(state: &mut State, control: usize, target: usize) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    let end = state.len() >> 2;
    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..end).for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;
            let (a, b, c, d) = unsafe {
                let a = *state.reals.get_unchecked(s0);
                let b = *state.imags.get_unchecked(s0);
                let c = *state.reals.get_unchecked(s1);
                let d = *state.imags.get_unchecked(s1);
                (a, b, c, d)
            };

            let a1 = SQRT_ONE_HALF * a;
            let b1 = SQRT_ONE_HALF * b;
            let c1 = SQRT_ONE_HALF * c;
            let d1 = SQRT_ONE_HALF * d;

            unsafe {
                *state.reals.get_unchecked_mut(s0) = a1 + c1;
                *state.imags.get_unchecked_mut(s0) = b1 + d1;
                *state.reals.get_unchecked_mut(s1) = a1 - c1;
                *state.imags.get_unchecked_mut(s1) = b1 - d1;
            }
        });
    } else {
        (0..end).into_par_iter().for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;
            let (a, b, c, d) = unsafe {
                let a = *state_re.get().add(s0);
                let b = *state_im.get().add(s0);
                let c = *state_re.get().add(s1);
                let d = *state_im.get().add(s1);
                (a, b, c, d)
            };

            let a1 = SQRT_ONE_HALF * a;
            let b1 = SQRT_ONE_HALF * b;
            let c1 = SQRT_ONE_HALF * c;
            let d1 = SQRT_ONE_HALF * d;

            unsafe {
                *state_re.get().add(s0) = a1 + c1;
                *state_im.get().add(s0) = b1 + d1;
                *state_re.get().add(s1) = a1 - c1;
                *state_im.get().add(s1) = b1 - d1;
            }
        });
    }
}

fn rx_apply_target_0(
    state_re: SendPtr<Float>,
    state_im: SendPtr<Float>,
    l: usize,
    cos: Float,
    neg_sin: Float,
) {
    unsafe {
        // a + ib
        let a = *state_re.get().add(l);
        let b = *state_im.get().add(l);

        // c + id
        let c = *state_re.get().add(l + 1);
        let d = *state_im.get().add(l + 1);

        *state_re.get().add(l) = a * cos - d * neg_sin;
        *state_im.get().add(l) = b * cos + c * neg_sin;

        *state_re.get().add(l + 1) = b * -neg_sin + c * cos;
        *state_im.get().add(l + 1) = d * cos + a * neg_sin;
    }
}

fn rx_apply_target(
    state_re: SendPtr<Float>,
    state_im: SendPtr<Float>,
    s0: usize,
    s1: usize,
    cos: Float,
    neg_sin: Float,
) {
    unsafe {
        // a + ib
        let a = *state_re.get().add(s0);
        let b = *state_im.get().add(s0);

        // c + id
        let c = *state_re.get().add(s1);
        let d = *state_im.get().add(s1);

        *state_re.get().add(s0) = a * cos - d * neg_sin;
        *state_im.get().add(s0) = b * cos + c * neg_sin;

        *state_re.get().add(s1) = b * -neg_sin + c * cos;
        *state_im.get().add(s1) = d * cos + a * neg_sin;
    }
}

fn rx_proc_chunk(
    state_re: SendPtr<Float>,
    state_im: SendPtr<Float>,
    chunk: usize,
    target: usize,
    cos: Float,
    neg_sin: Float,
) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let s0 = base + i;
        let s1 = s0 + dist;
        rx_apply_target(state_re, state_im, s0, s1, cos, neg_sin);
    }
}

fn rx_apply(state: &mut State, target: usize, angle: Float) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    let theta = angle * 0.5;
    let ct = Float::cos(theta);
    let nst = -Float::sin(theta);

    if target == 0 {
        if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
            (0..state.len()).step_by(2).for_each(|l| {
                rx_apply_target_0(state_re, state_im, l, ct, nst);
            })
        } else {
            (0..state.len()).into_par_iter().step_by(2).for_each(|l| {
                rx_apply_target_0(state_re, state_im, l, ct, nst);
            })
        }
    } else {
        let end = state.len() >> 1;
        let chunks = end >> target;
        if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
            (0..chunks).for_each(|chunk| {
                rx_proc_chunk(state_re, state_im, chunk, target, ct, nst);
            });
        } else {
            (0..chunks).into_par_iter().for_each(|chunk| {
                rx_proc_chunk(state_re, state_im, chunk, target, ct, nst);
            });
        }
    }
}

fn rx_c_apply(state: &mut State, control: usize, target: usize, angle: Float) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    let theta = angle * 0.5;
    let ct = Float::cos(theta);
    let nst = -Float::sin(theta);
    let end = state.len() >> 2;

    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..end).for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;
            rx_apply_target(state_re, state_im, s0, s1, ct, nst);
        });
    } else {
        (0..end).into_par_iter().for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;
            rx_apply_target(state_re, state_im, s0, s1, ct, nst);
        });
    }
}

fn rx_mc_apply(state: &mut State, mask: u64, target: usize, angle: Float) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    let theta = angle * 0.5;
    let ct = Float::cos(theta);
    let nst = -Float::sin(theta);

    let mut i = 0;
    let dist = 1 << target;

    while i < state.len() {
        if (i as u64 & mask) == mask {
            let s0 = i;
            let s1 = i + dist;
            rx_apply_target(state_re, state_im, s0, s1, ct, nst);
            i += dist;
        }
        i += 1;
    }
}

fn p_proc_chunk(state: &mut State, chunk: usize, target: usize, cos: Float, sin: Float) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;

    // s1 = base + i + dist, for i \in \{0, 1, 2, \ldots, dist-1\}
    for s1 in (base + dist)..(base + 2 * dist) {
        let z_re = state.reals[s1];
        let z_im = state.imags[s1];
        state.reals[s1] = z_re.mul_add(cos, -z_im * sin);
        state.imags[s1] = z_im.mul_add(cos, z_re * sin);
    }
}

fn p_proc_chunk_par(
    state_re: SendPtr<Float>,
    state_im: SendPtr<Float>,
    chunk: usize,
    target: usize,
    cos: Float,
    sin: Float,
) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;

    // s1 = base + i + dist, for i \in \{0, 1, 2, \ldots, dist-1\}
    for s1 in (base + dist)..(base + 2 * dist) {
        unsafe {
            let z_re = *state_re.get().add(s1);
            let z_im = *state_im.get().add(s1);
            *state_re.get().add(s1) = z_re.mul_add(cos, -z_im * sin);
            *state_im.get().add(s1) = z_im.mul_add(cos, z_re * sin);
        }
    }
}

fn p_apply(state: &mut State, target: usize, angle: Float) {
    let (sin, cos) = Float::sin_cos(angle);
    let end = state.len() >> 1;
    let chunks = end >> target;

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..chunks).for_each(|chunk| {
            p_proc_chunk(state, chunk, target, cos, sin);
        });
    } else {
        let state_re = SendPtr(state.reals.as_mut_ptr());
        let state_im = SendPtr(state.imags.as_mut_ptr());
        (0..chunks).into_par_iter().for_each(|chunk| {
            p_proc_chunk_par(state_re, state_im, chunk, target, cos, sin);
        });
    }
}

fn p_c_apply(state: &mut State, control: usize, target: usize, angle: Float) {
    let (sin, cos) = Float::sin_cos(angle);
    let end = state.len() >> 2;
    let marks = (target.min(control), target.max(control));

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        for i in 0..end {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let z_re = state.reals[s1];
            let z_im = state.imags[s1];
            state.reals[s1] = z_re.mul_add(cos, -z_im * sin);
            state.imags[s1] = z_im.mul_add(cos, z_re * sin);
        }
    } else {
        let state_re = SendPtr(state.reals.as_mut_ptr());
        let state_im = SendPtr(state.imags.as_mut_ptr());
        (0..end).into_par_iter().for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            unsafe {
                let z_re = *state_re.get().add(s1);
                let z_im = *state_im.get().add(s1);
                *state_re.get().add(s1) = z_re.mul_add(cos, -z_im * sin);
                *state_im.get().add(s1) = z_im.mul_add(cos, z_re * sin);
            }
        });
    }
}

fn p_mc_apply(state: &mut State, mask: u64, target: usize, theta: Float) {
    let (sin, cos) = Float::sin_cos(theta);
    let mut i = 0;
    let dist = 1 << target;

    while i < state.len() {
        if (i as u64 & mask) == mask {
            let s1 = i + dist;
            let z_re = state.reals[s1];
            let z_im = state.imags[s1];
            state.reals[s1] = z_re.mul_add(cos, -z_im * sin);
            state.imags[s1] = z_im.mul_add(cos, z_re * sin);
            i += dist;
        }
        i += 1;
    }
}

fn rz_apply_strategy1(state: &mut State, target: usize, diag_matrix: &[Amplitude; 2]) {
    let chunk_size = 1 << target;

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        state
            .reals
            .chunks_exact_mut(chunk_size)
            .zip(state.imags.chunks_exact_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (c0, c1))| {
                c0.iter_mut().zip(c1.iter_mut()).for_each(|(a, b)| {
                    let m = diag_matrix[i & 1];
                    let c = *a;
                    let d = *b;
                    *a = c.mul_add(m.re, -d * m.im);
                    *b = c.mul_add(m.im, d * m.re);
                });
            });
    } else {
        state
            .reals
            .par_chunks_exact_mut(chunk_size)
            .zip(state.imags.par_chunks_exact_mut(chunk_size))
            .enumerate()
            .with_min_len(1 << 11)
            .for_each(|(i, (c0, c1))| {
                c0.iter_mut().zip(c1.iter_mut()).for_each(|(a, b)| {
                    let m = diag_matrix[i & 1];
                    let c = *a;
                    let d = *b;
                    *a = c.mul_add(m.re, -d * m.im);
                    *b = c.mul_add(m.im, d * m.re);
                });
            });
    }
}

// NOTE: since we are checking pairs, rather than generating, we need to go through the *entire*
// state
fn rz_apply(state: &mut State, target: usize, angle: Float) {
    let theta = angle * 0.5;
    let (s, c) = Float::sin_cos(theta);
    let d0 = Amplitude { re: c, im: -s };
    let d1 = Amplitude { re: c, im: s };
    let diag_matrix = [d0, d1];
    rz_apply_strategy1(state, target, &diag_matrix);
}

fn rz_c_apply(state: &mut State, control: usize, target: usize, angle: Float) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    let theta = angle * 0.5;
    let (s, c) = Float::sin_cos(theta);
    let d0 = Amplitude { re: c, im: -s };
    let d1 = Amplitude { re: c, im: s };
    let diag_matrix = [d0, d1];
    let end = state.len() >> 2;

    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..end).for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;

            let m = diag_matrix[0];
            unsafe {
                let a = state_re.get().add(s0);
                let b = state_im.get().add(s0);
                let c = *a;
                let d = *b;

                *a = c.mul_add(m.re, -d * m.im);
                *b = c.mul_add(m.im, d * m.re);
            }
            let m = diag_matrix[1];
            unsafe {
                let a = state_re.get().add(s1);
                let b = state_im.get().add(s1);
                let c = *a;
                let d = *b;

                *a = c.mul_add(m.re, -d * m.im);
                *b = c.mul_add(m.im, d * m.re);
            }
        });
    } else {
        (0..end).into_par_iter().for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;
            let m = diag_matrix[0];
            unsafe {
                let a = state_re.get().add(s0);
                let b = state_im.get().add(s0);
                let c = *a;
                let d = *b;

                *a = c.mul_add(m.re, -d * m.im);
                *b = c.mul_add(m.im, d * m.re);
            }
            let m = diag_matrix[1];
            unsafe {
                let a = state_re.get().add(s1);
                let b = state_im.get().add(s1);
                let c = *a;
                let d = *b;

                *a = c.mul_add(m.re, -d * m.im);
                *b = c.mul_add(m.im, d * m.re);
            }
        });
    }
}

fn z_proc_chunk(state: &mut State, chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in base + dist..base + 2 * dist {
        unsafe {
            *state.reals.get_unchecked_mut(i) = -state.reals.get_unchecked(i);
            *state.imags.get_unchecked_mut(i) = -state.imags.get_unchecked(i);
        };
    }
}

fn z_proc_chunk_par(
    state_re: SendPtr<Float>,
    state_im: SendPtr<Float>,
    chunk: usize,
    target: usize,
) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in base + dist..base + 2 * dist {
        unsafe {
            *state_re.get().add(i) = -(*state_re.get().add(i));
            *state_im.get().add(i) = -(*state_im.get().add(i));
        };
    }
}

pub(crate) fn z_apply(state: &mut State, target: usize) {
    // NOTE: chunks == end >> target, where end == state.len() >> 1
    let chunks = state.len() >> (target + 1);

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..chunks).for_each(|c| z_proc_chunk(state, c, target));
    } else {
        let state_re = SendPtr(state.reals.as_mut_ptr());
        let state_im = SendPtr(state.imags.as_mut_ptr());
        (0..chunks)
            .into_par_iter()
            .for_each(|chunk| z_proc_chunk_par(state_re, state_im, chunk, target));
    }
}

fn ry_apply_strategy2(
    state_re: SendPtr<Float>,
    state_im: SendPtr<Float>,
    chunk: usize,
    target: usize,
    sin: Float,
    cos: Float,
) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let s0 = base + i;
        let s1 = s0 + dist;

        let (a, b, c, d) = unsafe {
            // a + ib
            let a = *state_re.get().add(s0);
            let b = *state_im.get().add(s0);

            // c + id
            let c = *state_re.get().add(s1);
            let d = *state_im.get().add(s1);
            (a, b, c, d)
        };

        unsafe {
            *state_re.get().add(s0) = a * cos - c * sin;
            *state_im.get().add(s0) = b * cos - d * sin;
            *state_re.get().add(s1) = a * sin + c * cos;
            *state_im.get().add(s1) = b * sin + d * cos;
        }
    }
}

fn ry_apply(state: &mut State, target: usize, angle: Float) {
    let theta = angle * 0.5;
    let (sin, cos) = Float::sin_cos(theta);

    let end = state.len() >> 1;
    let chunks = end >> target;
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..chunks).for_each(|chunk| {
            ry_apply_strategy2(state_re, state_im, chunk, target, sin, cos);
        });
    } else {
        (0..chunks).into_par_iter().for_each(|chunk| {
            ry_apply_strategy2(state_re, state_im, chunk, target, sin, cos);
        });
    }
}

fn ry_c_apply_strategy3(state: &mut State, control: usize, target: usize, angle: Float) {
    let theta = angle * 0.5;
    let (sin, cos) = Float::sin_cos(theta);
    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));
    let end = state.len() >> 2;

    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        for i in 0..end {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;

            let (a, b, c, d) = unsafe {
                // a + ib
                let a = *state_re.get().add(s0);
                let b = *state_im.get().add(s0);

                // c + id
                let c = *state_re.get().add(s1);
                let d = *state_im.get().add(s1);
                (a, b, c, d)
            };

            unsafe {
                *state_re.get().add(s0) = a * cos - c * sin;
                *state_im.get().add(s0) = b * cos - d * sin;
                *state_re.get().add(s1) = a * sin + c * cos;
                *state_im.get().add(s1) = b * sin + d * cos;
            }
        }
    } else {
        (0..end).into_par_iter().for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let s1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let s0 = s1 - dist;

            let (a, b, c, d) = unsafe {
                // a + ib
                let a = *state_re.get().add(s0);
                let b = *state_im.get().add(s0);

                // c + id
                let c = *state_re.get().add(s1);
                let d = *state_im.get().add(s1);
                (a, b, c, d)
            };

            unsafe {
                *state_re.get().add(s0) = a * cos - c * sin;
                *state_im.get().add(s0) = b * cos - d * sin;
                *state_re.get().add(s1) = a * sin + c * cos;
                *state_im.get().add(s1) = b * sin + d * cos;
            }
        });
    }
}

fn ry_c_apply(state: &mut State, control: usize, target: usize, angle: Float) {
    ry_c_apply_strategy3(state, control, target, angle);
}

fn ry_mc_apply(state: &mut State, mask: u64, target: usize, angle: Float) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    let theta = angle * 0.5;
    let ct = Float::cos(theta);
    let nst = -Float::sin(theta);

    let mut i = 0;
    let dist = 1 << target;

    while i < state.len() {
        if (i as u64 & mask) == mask {
            let s0 = i;
            let s1 = i + dist;
            rx_apply_target(state_re, state_im, s0, s1, ct, nst);
            i += dist;
        }
        i += 1;
    }
}

fn u_apply_target(
    state_re: SendPtr<Float>,
    state_im: SendPtr<Float>,
    g: &[Amplitude; 4],
    s0: usize,
    s1: usize,
) {
    let (c, d, m, n) = unsafe {
        let z0_re = *state_re.get().add(s0);
        let z0_im = *state_im.get().add(s0);
        let z1_re = *state_re.get().add(s1);
        let z1_im = *state_im.get().add(s1);
        (z0_re, z0_im, z1_re, z1_im)
    };

    let a = g[0].re;
    // let b = g[0].im;
    let k = g[1].re;
    let l = g[1].im;

    let q = g[2].re;
    let r = g[2].im;
    let s = g[3].re;
    let t = g[3].im;

    let t0 = a.mul_add(c, k.mul_add(m, -l * n));
    let t1 = a.mul_add(d, k.mul_add(n, l * m));
    let t2 = q.mul_add(c, (-r).mul_add(d, s.mul_add(m, -t * n)));
    let t3 = q.mul_add(d, r.mul_add(c, s.mul_add(n, t * m)));

    unsafe {
        *state_re.get().add(s0) = t0;
        *state_im.get().add(s0) = t1;
        *state_re.get().add(s1) = t2;
        *state_im.get().add(s1) = t3;
    }
}

fn u_apply_strategy2(
    state_re: SendPtr<Float>,
    state_im: SendPtr<Float>,
    g: &[Amplitude; 4],
    chunk: usize,
    target: usize,
) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let s0 = base + i;
        let s1 = s0 + dist;
        u_apply_target(state_re, state_im, g, s0, s1);
    }
}

fn u_apply(state: &mut State, target: usize, theta: Float, phi: Float, lambda: Float) {
    let (st, ct) = Float::sin_cos(theta * 0.5);
    let (sl, cl) = Float::sin_cos(lambda);
    let (spl, cpl) = Float::sin_cos(phi + lambda);
    let (sp, cp) = Float::sin_cos(phi);

    let c = Amplitude { re: ct, im: 0.0 };
    let ncs = Amplitude {
        re: -cl * st,
        im: -sl * st,
    };
    let es = Amplitude {
        re: cp * st,
        im: sp * st,
    };
    let ec = Amplitude {
        re: cpl * ct,
        im: spl * ct,
    };
    let g = [c, ncs, es, ec];

    // NOTE: chunks == end >> target, where end == state.len() >> 1
    let chunks = state.len() >> (target + 1);
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    if Config::global().threads < 2 || state.n < LOW_QUBIT_THRESHOLD {
        (0..chunks).for_each(|chunk| {
            u_apply_strategy2(state_re, state_im, &g, chunk, target);
        });
    } else {
        (0..chunks).into_par_iter().for_each(|chunk| {
            u_apply_strategy2(state_re, state_im, &g, chunk, target);
        });
    }
}

fn u_c_apply(
    state: &mut State,
    theta: Float,
    phi: Float,
    lambda: Float,
    control: usize,
    target: usize,
) {
    let (st, ct) = Float::sin_cos(theta * 0.5);
    let (sl, cl) = Float::sin_cos(lambda);
    let (spl, cpl) = Float::sin_cos(phi + lambda);
    let (sp, cp) = Float::sin_cos(phi);

    let c = Amplitude { re: ct, im: 0.0 };
    let ncs = Amplitude {
        re: -cl * st,
        im: -sl * st,
    };
    let es = Amplitude {
        re: cp * st,
        im: sp * st,
    };
    let ec = Amplitude {
        re: cpl * ct,
        im: spl * ct,
    };
    let g = [c, ncs, es, ec];

    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    let distance = 1 << target;
    let mask = (1 << control) | (1 << target);

    for i in 0..state.len() {
        if i & mask == mask {
            let s1 = i;
            let s0 = s1 - distance;
            u_apply_target(state_re, state_im, &g, s0, s1);
        }
    }
}

fn bit_flip_noise_apply(state: &mut State, prob: Float, target: usize) -> bool {
    let mut rng = rand::thread_rng();
    let epsilon: Float = rng.gen();

    if epsilon <= prob {
        x_apply(state, target);
        return true;
    }
    false
}

fn swap_apply(state: &mut State, t0: usize, t1: usize) {
    assert!(usize::from(state.n) > t0 && usize::from(state.n) > t1);

    for i in 0..state.len() {
        if ((i >> t0) & 1) == 0 && ((i >> t1) & 1) == 1 {
            let j = i + (1 << t0) - (1 << t1);
            state.reals.swap(i, j);
            state.imags.swap(i, j);
        }
    }
}

fn unitary_col_vec_mul(
    unitary: &Unitary,
    vec_reals: &[Float],
    vec_imags: &[Float],
) -> (Vec<Float>, Vec<Float>) {
    assert!(
        unitary.height == unitary.width
            && vec_reals.len() == unitary.height
            && vec_imags.len() == unitary.height
    );
    let chunk_size = unitary.width;

    let mut reals = Vec::with_capacity(vec_reals.len());
    let mut imags = Vec::with_capacity(vec_imags.len());

    // TODO(saveliy): parallelize this
    unitary
        .reals
        .chunks_exact(chunk_size)
        .zip(unitary.imags.chunks_exact(chunk_size))
        .for_each(|(row_reals, row_imags)| {
            let mut dot_prod_re = 0.0;
            let mut dot_prod_im = 0.0;
            row_reals
                .iter()
                .zip(row_imags.iter())
                .zip(vec_reals.iter())
                .zip(vec_imags.iter())
                .for_each(|(((a, b), s_re), s_im)| {
                    dot_prod_re += *a * s_re - *b * s_im;
                    dot_prod_im += *a * s_im + *b * s_re;
                });
            reals.push(dot_prod_re);
            imags.push(dot_prod_im);
        });
    (reals, imags)
}

/// Apply a Unitary to the `State`
pub fn transform_u(state: &mut State, u: &Unitary, t: usize) {
    assert_eq!(u.height, u.width);
    let m: usize = usize::try_from(u.height.ilog2()).unwrap();
    let n: usize = usize::try_from(u.width.ilog2()).unwrap();

    let mut vec_reals = vec![0.0; 1 << m];
    let mut vec_imags = vec![0.0; 1 << m];

    for suffix in 0..(1 << t) {
        for prefix in 0..(1 << (n - m - t)) {
            for target in 0..(1 << m) {
                let k = prefix * (1 << (t + m)) + target * (1 << t) + suffix;
                vec_reals[target] = state.reals[k];
                vec_imags[target] = state.imags[k];
            }

            let (vec_reals_out, vec_imags_out) = unitary_col_vec_mul(u, &vec_reals, &vec_imags);

            for target in 0..(1 << m) {
                let k = prefix * (1 << (t + m)) + target * (1 << t) + suffix;
                state.reals[k] = vec_reals_out[target];
                state.imags[k] = vec_imags_out[target];
            }
        }
    }
}

/// Apply a controlled Unitary to the `State`
pub fn c_transform_u(state: &mut State, u: &Unitary, c: usize, t: usize) {
    assert_eq!(u.height, u.width);
    let m: usize = usize::try_from(u.height.ilog2()).unwrap();
    let n: usize = usize::try_from(u.width.ilog2()).unwrap();

    let mut vec_reals = vec![0.0; 1 << m];
    let mut vec_imags = vec![0.0; 1 << m];
    let mut targets = Vec::new();

    for suffix in 0..(1 << t) {
        for prefix in 0..(1 << (n - m - t)) {
            targets.clear();
            for idx in 0..(1 << m) {
                let k = prefix * (1 << (t + m)) + idx * (1 << t) + suffix;
                if ((k >> c) & 1) == 1 {
                    vec_reals[idx] = state.reals[k];
                    vec_imags[idx] = state.imags[k];
                    targets.push(k)
                }
            }

            let (vec_reals_out, vec_imags_out) = unitary_col_vec_mul(u, &vec_reals, &vec_imags);

            for idx in 0..(1 << m) {
                let k = prefix * (1 << (t + m)) + idx * (1 << t) + suffix;
                if ((k >> c) & 1) == 1 {
                    state.reals[k] = vec_reals_out[idx];
                    state.imags[k] = vec_imags_out[idx];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        core::iqft,
        math::{pow2f, PI},
        utils::{assert_float_closeness, gen_random_state, mat_mul_2x2, swap},
    };

    use super::*;

    fn qcbm_functional(n: usize) -> State {
        let mut state = State::new(n);
        let pairs: Vec<_> = (0..n).map(|i| (i, (i + 1) % n)).collect();

        for i in 0..n {
            apply(Gate::RX(1.0), &mut state, i);
            apply(Gate::RZ(1.0), &mut state, i);
        }

        for (p0, p1) in pairs.iter().take(n - 1) {
            c_apply(Gate::X, &mut state, *p0, *p1);
        }

        for _ in 0..9 {
            for i in 0..n {
                apply(Gate::RZ(1.0), &mut state, i);
                apply(Gate::RX(1.0), &mut state, i);
                apply(Gate::RZ(1.0), &mut state, i);
            }

            for (p0, p1) in pairs.iter().take(n - 1) {
                c_apply(Gate::X, &mut state, *p0, *p1);
            }
        }

        for i in 0..n {
            apply(Gate::RZ(1.0), &mut state, i);
            apply(Gate::RX(1.0), &mut state, i);
        }
        state
    }

    #[test]
    fn h_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::H, &mut state, i);
        }

        for i in 0..(1 << n) {
            assert_float_closeness(state.reals[i], 0.35355339059327384, 1e-10);
            assert_float_closeness(state.imags[i], 0.0, 1e-10);
        }
    }

    #[test]
    fn x_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::X, &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.0, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 1.0, 1e-10);
        assert_float_closeness(state.imags[7], 0.0, 1e-10);
    }

    #[test]
    fn x_gate_20_qubits() {
        let n = 20;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::X, &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.0, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[(1 << n) - 1], 1.0, 1e-10);
        assert_float_closeness(state.imags[(1 << n) - 1], 0.0, 1e-10);
    }

    #[test]
    fn y_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::Y, &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.0, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 0.0, 1e-10);
        assert_float_closeness(state.imags[7], -1.0, 1e-10);
    }

    #[test]
    fn y_gate_20_qubits() {
        let n = 20;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::Y, &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.0, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[(1 << n) - 1], 1.0, 1e-10);
        assert_float_closeness(state.imags[(1 << n) - 1], 0.0, 1e-10);
    }

    #[test]
    fn z_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::Z, &mut state, i);
        }

        for i in 0..(1 << n) {
            if i == 0 {
                assert_float_closeness(state.reals[i], 1.0, 1e-10);
                assert_float_closeness(state.imags[i], 0.0, 1e-10);
            } else {
                assert_float_closeness(state.reals[i], 0.0, 1e-10);
                assert_float_closeness(state.imags[i], 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn p_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::P(PI), &mut state, i);
        }

        for i in 0..(1 << n) {
            if i == 0 {
                assert_float_closeness(state.reals[i], 1.0, 1e-10);
                assert_float_closeness(state.imags[i], 0.0, 1e-10);
            } else {
                assert_float_closeness(state.reals[i], 0.0, 1e-10);
                assert_float_closeness(state.imags[i], 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn rx_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::RX(1.0), &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.6758712218347053, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[1], 0.0, 1e-10);
        assert_float_closeness(state.imags[1], -0.3692301313020644, 1e-10);

        assert_float_closeness(state.reals[2], 0.0, 1e-10);
        assert_float_closeness(state.imags[2], -0.3692301313020644, 1e-10);

        assert_float_closeness(state.reals[3], -0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[3], 0.0, 1e-10);

        assert_float_closeness(state.reals[4], 0.0, 1e-10);
        assert_float_closeness(state.imags[4], -0.3692301313020644, 1e-10);

        assert_float_closeness(state.reals[5], -0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[5], 0.0, 1e-10);

        assert_float_closeness(state.reals[6], -0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[6], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 0.0, 1e-10);
        assert_float_closeness(state.imags[7], 0.11019540730213864, 1e-10);
    }

    #[test]
    fn ry_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::RY(1.0), &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.6758712218347053, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[1], 0.3692301313020644, 1e-10);
        assert_float_closeness(state.imags[1], 0.0, 1e-10);

        assert_float_closeness(state.reals[2], 0.3692301313020644, 1e-10);
        assert_float_closeness(state.imags[2], 0.0, 1e-10);

        assert_float_closeness(state.reals[3], 0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[3], 0.0, 1e-10);

        assert_float_closeness(state.reals[4], 0.3692301313020644, 1e-10);
        assert_float_closeness(state.imags[4], 0.0, 1e-10);

        assert_float_closeness(state.reals[5], 0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[5], 0.0, 1e-10);

        assert_float_closeness(state.reals[6], 0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[6], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 0.11019540730213864, 1e-10);
        assert_float_closeness(state.imags[7], 0.0, 1e-10);
    }

    #[test]
    fn rz_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::RZ(1.0), &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.07073720166770296, 1e-10);
        assert_float_closeness(state.imags[0], -0.9974949866040546, 1e-10);

        assert_float_closeness(state.reals[1], 0.0, 1e-10);
        assert_float_closeness(state.imags[1], 0.0, 1e-10);

        assert_float_closeness(state.reals[2], 0.0, 1e-10);
        assert_float_closeness(state.imags[2], 0.0, 1e-10);

        assert_float_closeness(state.reals[3], 0.0, 1e-10);
        assert_float_closeness(state.imags[3], 0.0, 1e-10);

        assert_float_closeness(state.reals[4], 0.0, 1e-10);
        assert_float_closeness(state.imags[4], 0.0, 1e-10);

        assert_float_closeness(state.reals[5], 0.0, 1e-10);
        assert_float_closeness(state.imags[5], 0.0, 1e-10);

        assert_float_closeness(state.reals[6], 0.0, 1e-10);
        assert_float_closeness(state.imags[6], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 0.0, 1e-10);
        assert_float_closeness(state.imags[7], 0.0, 1e-10);
    }

    #[test]
    fn value_encoding_3_qubits() {
        let v = 2.4;
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::H, &mut state, i);
        }
        for i in 0..n {
            apply(Gate::P(2.0 * PI / (pow2f(i + 1)) * v), &mut state, i);
        }
        let targets: Vec<usize> = (0..n).rev().collect();

        iqft(&mut state, &targets);
    }

    #[test]
    fn value_encoding_20_qubits() {
        let v = 2.4;
        let n = 20;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::H, &mut state, i);
        }
        for i in 0..n {
            apply(Gate::P(2.0 * PI / (pow2f(i + 1)) * v), &mut state, i);
        }
        let targets: Vec<usize> = (0..n).rev().collect();

        iqft(&mut state, &targets);
    }

    #[test]
    fn qcbm_3_qubits() {
        let state = qcbm_functional(3);

        assert_float_closeness(state.reals[0], 0.18037770683997864, 1e-10);
        assert_float_closeness(state.imags[0], -0.17626993141958947, 1e-10);

        assert_float_closeness(state.reals[7], 0.014503954556966365, 1e-10);
        assert_float_closeness(state.imags[7], -0.11198008105074927, 1e-10);
    }

    #[test]
    fn qcbm_20_qubits() {
        let state = qcbm_functional(20);

        assert_float_closeness(state.reals[0], -0.0022221321676945643, 1e-10);
        assert_float_closeness(state.imags[0], 0.001743068112560825, 1e-10);

        assert_float_closeness(state.reals[7], -0.0031017461877124453, 1e-10);
        assert_float_closeness(state.imags[7], -0.0034043237120339686, 1e-10);

        assert_float_closeness(state.reals[12], 0.0005494086136357235, 1e-10);
        assert_float_closeness(state.imags[12], -0.00009827749580581964, 1e-10);
    }

    #[test]
    fn u_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::U(1.0, 1.0, 1.0), &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.6758712218347053, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[1], 0.19949589133850137, 1e-10);
        assert_float_closeness(state.imags[1], 0.3106964422074971, 1e-10);

        assert_float_closeness(state.reals[2], 0.19949589133850137, 1e-10);
        assert_float_closeness(state.imags[2], 0.3106964422074971, 1e-10);

        assert_float_closeness(state.reals[3], -0.08394153605985091, 1e-10);
        assert_float_closeness(state.imags[3], 0.18341560247417849, 1e-10);

        assert_float_closeness(state.reals[2], 0.19949589133850137, 1e-10);
        assert_float_closeness(state.imags[2], 0.3106964422074971, 1e-10);

        assert_float_closeness(state.reals[3], -0.08394153605985091, 1e-10);
        assert_float_closeness(state.imags[3], 0.18341560247417849, 1e-10);

        assert_float_closeness(state.reals[3], -0.08394153605985091, 1e-10);
        assert_float_closeness(state.imags[3], 0.18341560247417849, 1e-10);

        assert_float_closeness(state.reals[7], -0.1090926263889472, 1e-10);
        assert_float_closeness(state.imags[7], 0.015550776766638148, 1e-10);
    }

    #[test]
    fn u_gate_1_qubit() {
        let mut state = State::new(1);

        let lambda = 1.0;
        let theta = 2.0;
        let phi = 3.0;

        apply(Gate::U(theta, phi, lambda), &mut state, 0);
        assert_float_closeness(state.reals[0], 0.5403023058681398, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[1], -0.833049961066805, 1e-10);
        assert_float_closeness(state.imags[1], 0.11874839215823475, 1e-10);

        // Check other base vector
        let mut state = State::new(1);
        apply(Gate::X, &mut state, 0);
        apply(Gate::U(theta, phi, lambda), &mut state, 0);

        assert_float_closeness(state.reals[0], -0.4546487134128409, 1e-10);
        assert_float_closeness(state.imags[0], -0.7080734182735712, 1e-10);

        assert_float_closeness(state.reals[1], -0.35316515556860967, 1e-10);
        assert_float_closeness(state.imags[1], -0.4089021333016357, 1e-10);
    }

    #[test]
    fn swap_9_qubits() {
        const N: usize = 9;
        let mut state_0 = gen_random_state(N);
        let mut state_1 = state_0.clone();

        let (t0, t1) = (0, 1);
        swap(&mut state_0, t0, t1);
        swap_apply(&mut state_1, t0, t1);

        assert_eq!(state_0.n, state_1.n);
        assert_eq!(state_0.reals, state_1.reals);
        assert_eq!(state_0.imags, state_1.imags);
    }

    #[test]
    fn swap_all_qubits() {
        const N: usize = 3;
        let mut state_0 = gen_random_state(N);
        let mut state_1 = state_0.clone();

        for i in 0..(N >> 1) {
            swap(&mut state_0, i, N - 1 - i);
            swap_apply(&mut state_1, i, N - 1 - i);
        }

        assert_eq!(state_0.n, state_1.n);
        assert_eq!(state_0.reals, state_1.reals);
        assert_eq!(state_0.imags, state_1.imags);
    }

    #[test]
    fn h_inverse() {
        let h = Gate::H.to_matrix();
        let h_inv = Gate::H.inverse().to_matrix();

        let identity = mat_mul_2x2(h, h_inv);
        assert_float_closeness(identity[0].re, 1.0, 0.001);
        assert_float_closeness(identity[0].im, 0.0, 0.001);
        assert_float_closeness(identity[1].re, 0.0, 0.001);
        assert_float_closeness(identity[1].im, 0.0, 0.001);
        assert_float_closeness(identity[2].re, 0.0, 0.001);
        assert_float_closeness(identity[2].im, 0.0, 0.001);
        assert_float_closeness(identity[3].re, 1.0, 0.001);
        assert_float_closeness(identity[3].im, 0.0, 0.001);
    }

    #[test]
    fn x_inverse() {
        let x = Gate::X.to_matrix();
        let x_inv = Gate::X.inverse().to_matrix();

        let identity = mat_mul_2x2(x, x_inv);
        assert_float_closeness(identity[0].re, 1.0, 0.001);
        assert_float_closeness(identity[0].im, 0.0, 0.001);
        assert_float_closeness(identity[1].re, 0.0, 0.001);
        assert_float_closeness(identity[1].im, 0.0, 0.001);
        assert_float_closeness(identity[2].re, 0.0, 0.001);
        assert_float_closeness(identity[2].im, 0.0, 0.001);
        assert_float_closeness(identity[3].re, 1.0, 0.001);
        assert_float_closeness(identity[3].im, 0.0, 0.001);
    }

    #[test]
    fn y_inverse() {
        let y = Gate::Y.to_matrix();
        let y_inv = Gate::Y.inverse().to_matrix();

        let identity = mat_mul_2x2(y, y_inv);
        assert_float_closeness(identity[0].re, 1.0, 0.001);
        assert_float_closeness(identity[0].im, 0.0, 0.001);
        assert_float_closeness(identity[1].re, 0.0, 0.001);
        assert_float_closeness(identity[1].im, 0.0, 0.001);
        assert_float_closeness(identity[2].re, 0.0, 0.001);
        assert_float_closeness(identity[2].im, 0.0, 0.001);
        assert_float_closeness(identity[3].re, 1.0, 0.001);
        assert_float_closeness(identity[3].im, 0.0, 0.001);
    }

    #[test]
    fn z_inverse() {
        let z = Gate::Z.to_matrix();
        let z_inv = Gate::Z.inverse().to_matrix();

        let identity = mat_mul_2x2(z, z_inv);
        assert_float_closeness(identity[0].re, 1.0, 0.001);
        assert_float_closeness(identity[0].im, 0.0, 0.001);
        assert_float_closeness(identity[1].re, 0.0, 0.001);
        assert_float_closeness(identity[1].im, 0.0, 0.001);
        assert_float_closeness(identity[2].re, 0.0, 0.001);
        assert_float_closeness(identity[2].im, 0.0, 0.001);
        assert_float_closeness(identity[3].re, 1.0, 0.001);
        assert_float_closeness(identity[3].im, 0.0, 0.001);
    }

    #[test]
    fn p_inverse() {
        let p = Gate::P(2.03).to_matrix();
        let p_inv = Gate::P(2.03).inverse().to_matrix();

        let identity = mat_mul_2x2(p, p_inv);
        assert_float_closeness(identity[0].re, 1.0, 0.001);
        assert_float_closeness(identity[0].im, 0.0, 0.001);
        assert_float_closeness(identity[1].re, 0.0, 0.001);
        assert_float_closeness(identity[1].im, 0.0, 0.001);
        assert_float_closeness(identity[2].re, 0.0, 0.001);
        assert_float_closeness(identity[2].im, 0.0, 0.001);
        assert_float_closeness(identity[3].re, 1.0, 0.001);
        assert_float_closeness(identity[3].im, 0.0, 0.001);
    }

    #[test]
    fn rx_inverse() {
        let rx = Gate::RX(2.03).to_matrix();
        let rx_inv = Gate::RX(2.03).inverse().to_matrix();

        let identity = mat_mul_2x2(rx, rx_inv);
        assert_float_closeness(identity[0].re, 1.0, 0.001);
        assert_float_closeness(identity[0].im, 0.0, 0.001);
        assert_float_closeness(identity[1].re, 0.0, 0.001);
        assert_float_closeness(identity[1].im, 0.0, 0.001);
        assert_float_closeness(identity[2].re, 0.0, 0.001);
        assert_float_closeness(identity[2].im, 0.0, 0.001);
        assert_float_closeness(identity[3].re, 1.0, 0.001);
        assert_float_closeness(identity[3].im, 0.0, 0.001);
    }

    #[test]
    fn rz_inverse() {
        let rz = Gate::RZ(3.03).to_matrix();
        let rz_inv = Gate::RZ(3.03).inverse().to_matrix();

        let identity = mat_mul_2x2(rz, rz_inv);
        assert_float_closeness(identity[0].re, 1.0, 0.001);
        assert_float_closeness(identity[0].im, 0.0, 0.001);
        assert_float_closeness(identity[1].re, 0.0, 0.001);
        assert_float_closeness(identity[1].im, 0.0, 0.001);
        assert_float_closeness(identity[2].re, 0.0, 0.001);
        assert_float_closeness(identity[2].im, 0.0, 0.001);
        assert_float_closeness(identity[3].re, 1.0, 0.001);
        assert_float_closeness(identity[3].im, 0.0, 0.001);
    }

    #[test]
    fn ry_inverse() {
        let ry = Gate::RY(3.03).to_matrix();
        let ry_inv = Gate::RY(3.03).inverse().to_matrix();

        let identity = mat_mul_2x2(ry, ry_inv);
        assert_float_closeness(identity[0].re, 1.0, 0.001);
        assert_float_closeness(identity[0].im, 0.0, 0.001);
        assert_float_closeness(identity[1].re, 0.0, 0.001);
        assert_float_closeness(identity[1].im, 0.0, 0.001);
        assert_float_closeness(identity[2].re, 0.0, 0.001);
        assert_float_closeness(identity[2].im, 0.0, 0.001);
        assert_float_closeness(identity[3].re, 1.0, 0.001);
        assert_float_closeness(identity[3].im, 0.0, 0.001);
    }

    #[test]
    fn u_inverse() {
        let u = Gate::U(1.0, 2.0, 3.0).to_matrix();
        let u_inv = Gate::U(1.0, 2.0, 3.0).inverse().to_matrix();

        let identity = mat_mul_2x2(u, u_inv);
        assert_float_closeness(identity[0].re, 1.0, 0.001);
        assert_float_closeness(identity[0].im, 0.0, 0.001);
        assert_float_closeness(identity[1].re, 0.0, 0.001);
        assert_float_closeness(identity[1].im, 0.0, 0.001);
        assert_float_closeness(identity[2].re, 0.0, 0.001);
        assert_float_closeness(identity[2].im, 0.0, 0.001);
        assert_float_closeness(identity[3].re, 1.0, 0.001);
        assert_float_closeness(identity[3].im, 0.0, 0.001);
    }

    #[test]
    fn unitary_inverse() {}

    #[test]
    #[should_panic]
    fn m_inverse() {
        let _m = Gate::M;
        let _m_inv = Gate::M.inverse();
    }

    #[test]
    #[should_panic]
    fn swap_inverse() {
        let _swap = Gate::SWAP(0, 1);
        let _swap_inv = Gate::SWAP(0, 1).inverse().to_matrix();
    }

    #[test]
    fn unitary_column_vector_multiplication() {
        // Test case 1: Unitary matrix with real components
        let u1 = Unitary {
            height: 2,
            width: 2,
            reals: vec![1.0, 0.0, 0.0, 1.0],
            imags: vec![0.0, 0.0, 0.0, 0.0],
        };

        let vec_reals1 = vec![2.0, 3.0];
        let vec_imags1 = vec![4.0, 5.0];

        let (result_reals1, result_imags1) = unitary_col_vec_mul(&u1, &vec_reals1, &vec_imags1);

        assert_eq!(result_reals1, vec![2.0, 3.0]);
        assert_eq!(result_imags1, vec![4.0, 5.0]);

        // Test case 2: Unitary matrix with complex components
        let u2 = Unitary {
            height: 2,
            width: 2,
            reals: vec![0.0, 1.0, 1.0, 0.0],
            imags: vec![0.0, 0.0, 0.0, 0.0],
        };

        let vec_reals2 = vec![2.0, 3.0];
        let vec_imags2 = vec![4.0, 5.0];

        let (result_reals2, result_imags2) = unitary_col_vec_mul(&u2, &vec_reals2, &vec_imags2);

        assert_eq!(result_reals2, vec![3.0, 2.0]);
        assert_eq!(result_imags2, vec![5.0, 4.0]);
    }

    #[test]
    fn ch() {
        let n = 3;
        let mut state = State::new(n);

        for t in 0..n {
            apply(Gate::H, &mut state, t);
        }

        state
            .reals
            .iter()
            .zip(state.imags.iter())
            .for_each(|(z_re, z_im)| {
                assert_float_closeness(*z_re, 0.353553391, 0.0001);
                assert_float_closeness(*z_im, 0.0, 0.0001);
            });

        c_apply(Gate::H, &mut state, 0, 1);

        let mut i = 0;

        while i < state.len() - 3 {
            assert_float_closeness(state.reals[i], 0.353553391, 0.0001);
            assert_float_closeness(state.reals[i + 1], 0.5, 0.0001);
            assert_float_closeness(state.reals[i + 2], 0.353553391, 0.0001);
            assert_float_closeness(state.reals[i + 3], 0.0, 0.0001);

            assert_float_closeness(state.imags[i], 0.0, 0.0001);
            assert_float_closeness(state.imags[i + 1], 0.0, 0.0001);
            assert_float_closeness(state.imags[i + 2], 0.0, 0.0001);
            assert_float_closeness(state.imags[i + 3], 0.0, 0.0001);
            i += 4;
        }
    }

    #[test]
    fn crz() {
        let n = 3;
        let mut state = State::new(n);

        for t in 0..n {
            apply(Gate::H, &mut state, t);
        }

        state
            .reals
            .iter()
            .zip(state.imags.iter())
            .for_each(|(z_re, z_im)| {
                assert_float_closeness(*z_re, 0.353553391, 0.0001);
                assert_float_closeness(*z_im, 0.0, 0.0001);
            });

        c_apply(Gate::RZ(PI / 2.0), &mut state, 0, 1);

        let mut i = 0;

        while i < state.len() - 3 {
            assert_float_closeness(state.reals[i], 0.353553391, 0.0001);
            assert_float_closeness(state.imags[i], 0.0, 0.0001);

            assert_float_closeness(state.reals[i + 1], 0.25, 0.0001);
            assert_float_closeness(state.imags[i + 1], -0.25, 0.0001);

            assert_float_closeness(state.reals[i + 2], 0.353553391, 0.0001);
            assert_float_closeness(state.imags[i + 2], 0.0, 0.0001);

            assert_float_closeness(state.reals[i + 3], 0.25, 0.0001);
            assert_float_closeness(state.imags[i + 3], 0.25, 0.0001);
            i += 4;
        }
    }

    #[test]
    fn bit_flip_noise() {
        const N: usize = 1;
        let state0 = gen_random_state(N);
        let mut state1 = state0.clone();

        apply(Gate::BitFlipNoise(0.0), &mut state1, 0);
        assert_eq!(
            state0.reals, state1.reals,
            "BitFlipNoise with prob=0.0 should have no effect"
        );
        assert_eq!(
            state0.imags, state1.imags,
            "BitFlipNoise with prob=0.0 should have no effect"
        );

        // BitFlipNoise with prob=1.0 is equivalent to just applying the X gate to the provided target qubit
        apply(Gate::BitFlipNoise(1.0), &mut state1, 0);
        assert_float_closeness(state1.reals[0], state0.reals[1], 0.00001);
        assert_float_closeness(state1.reals[1], state0.reals[0], 0.00001);
        assert_float_closeness(state1.imags[0], state0.imags[1], 0.00001);
        assert_float_closeness(state1.imags[1], state0.imags[0], 0.00001);
    }

    #[test]
    fn controlled_u() {
        const N: usize = 3;
        let mut state = State::new(N);

        for target in 0..N {
            h_apply(&mut state, target);
        }
        u_c_apply(&mut state, 1.0, 2.0, 3.0, 0, 1);

        let expected_reals = [
            0.35355339, 0.47807852, 0.35355339, 0.01747458, 0.35355339, 0.47807852, 0.35355339,
            0.01747458,
        ];
        let expected_imags = [
            0.0,
            -0.0239202,
            0.0,
            -0.14339942,
            0.0,
            -0.0239202,
            0.0,
            -0.14339942,
        ];

        state
            .reals
            .iter()
            .zip(expected_reals.iter())
            .for_each(|(actual, expected)| {
                assert_float_closeness(*actual, *expected, 0.00001);
            });
        state
            .imags
            .iter()
            .zip(expected_imags.iter())
            .for_each(|(actual, expected)| {
                assert_float_closeness(*actual, *expected, 0.00001);
            });
    }
}
