//! Abstractions for a quantum logic gates
use crate::{
    config::Config,
    core::State,
    math::{Amplitude, Float, SQRT_ONE_HALF},
};
use rayon::prelude::*;

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
#[derive(Clone, Copy)]
pub enum Gate {
    /// Hadamard gate. See <https://en.wikipedia.org/wiki/Quantum_logic_gate#Hadamard_gate>
    H,
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
    /// General single qubit rotation. See <https://en.wikipedia.org/wiki/List_of_quantum_logic_gates#Other_named_gates>
    U((Float, Float, Float)),
}

/// Single Target, No Controls
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
        Gate::U((theta, phi, lambda)) => u_apply(state, target, theta, phi, lambda),
    }
}

/// Single Control, Single Target
pub fn c_apply(gate: Gate, state: &mut State, control: usize, target: usize) {
    match gate {
        Gate::X => x_c_apply(state, control, target),
        Gate::Y => y_c_apply(state, control, target),
        Gate::P(theta) => p_c_apply(state, control, target, theta),
        Gate::RX(theta) => rx_c_apply(state, control, target, theta),
        Gate::RY(theta) => ry_c_apply(state, control, target, theta),
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

fn x_apply_target_0(state_re: SendPtr<Float>, state_im: SendPtr<Float>, l0: usize) {
    unsafe {
        std::ptr::swap(state_re.get().add(l0), state_re.get().add(l0 + 1));
        std::ptr::swap(state_im.get().add(l0), state_im.get().add(l0 + 1));
    }
}

fn x_apply_target(state_re: SendPtr<Float>, state_im: SendPtr<Float>, l0: usize, l1: usize) {
    unsafe {
        std::ptr::swap(state_re.get().add(l0), state_re.get().add(l1));
        std::ptr::swap(state_im.get().add(l0), state_im.get().add(l1));
    }
}

fn x_proc_chunk(state_re: SendPtr<Float>, state_im: SendPtr<Float>, chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let l0 = base + i;
        let l1 = l0 + dist;
        x_apply_target(state_re, state_im, l0, l1)
    }
}

fn x_apply(state: &mut State, target: usize) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    if target == 0 {
        if Config::global().threads < 2 {
            (0..state.len()).step_by(2).for_each(|l0| {
                x_apply_target_0(state_re, state_im, l0);
            });
        } else {
            (0..state.len()).into_par_iter().step_by(2).for_each(|l0| {
                x_apply_target_0(state_re, state_im, l0);
            });
        }
    } else {
        let end = state.len() >> 1;
        let chunks = end >> target;

        if Config::global().threads < 2 {
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

    if Config::global().threads < 2 {
        (0..end).for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let l1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let l0 = l1 - dist;
            x_apply_target(state_re, state_im, l0, l1);
        });
    } else {
        (0..end).into_par_iter().for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let l1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let l0 = l1 - dist;
            x_apply_target(state_re, state_im, l0, l1);
        });
    }
}

fn x_cc_apply(state: &mut State, control0: usize, control1: usize, target: usize) {
    let mut i = 0;
    let dist = 1 << target;

    while i < state.len() {
        let c0c1_set = (((i >> control0) & 1) != 0) && (((i >> control1) & 1) != 0);
        if c0c1_set {
            let l0 = i;
            let l1 = i + dist;
            unsafe {
                let temp0 = *state.reals.get_unchecked(l0);
                *state.reals.get_unchecked_mut(l0) = *state.reals.get_unchecked(l1);
                *state.reals.get_unchecked_mut(l1) = temp0;

                let temp1 = *state.imags.get_unchecked(l0);
                *state.imags.get_unchecked_mut(l0) = *state.imags.get_unchecked(l1);
                *state.imags.get_unchecked_mut(l1) = temp1;
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

fn y_apply(state: &mut State, target: usize) {
    let end = state.len() >> 1;
    let chunks = end >> target;

    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    if Config::global().threads < 2 {
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

    if Config::global().threads < 2 {
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

/// Apply H gate via Concatenation strategy (i.e., Strategy 2)
fn h_apply_concat(state: &mut State, chunk: usize, target: usize) {
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

fn h_apply_concat_par(
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

    if Config::global().threads < 2 {
        (0..chunks).for_each(|c| {
            h_apply_concat(state, c, target);
        });
    } else {
        let state_re = SendPtr(state.reals.as_mut_ptr());
        let state_im = SendPtr(state.imags.as_mut_ptr());

        (0..chunks).into_par_iter().for_each(|c| {
            h_apply_concat_par(state_re, state_im, c, target);
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
    l0: usize,
    l1: usize,
    cos: Float,
    neg_sin: Float,
) {
    unsafe {
        // a + ib
        let a = *state_re.get().add(l0);
        let b = *state_im.get().add(l0);

        // c + id
        let c = *state_re.get().add(l1);
        let d = *state_im.get().add(l1);

        *state_re.get().add(l0) = a * cos - d * neg_sin;
        *state_im.get().add(l0) = b * cos + c * neg_sin;

        *state_re.get().add(l1) = b * -neg_sin + c * cos;
        *state_im.get().add(l1) = d * cos + a * neg_sin;
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
        let l0 = base + i;
        let l1 = l0 + dist;
        rx_apply_target(state_re, state_im, l0, l1, cos, neg_sin);
    }
}

fn rx_apply(state: &mut State, target: usize, angle: Float) {
    let state_re = SendPtr(state.reals.as_mut_ptr());
    let state_im = SendPtr(state.imags.as_mut_ptr());

    let theta = angle * 0.5;
    let ct = Float::cos(theta);
    let nst = -Float::sin(theta);

    if target == 0 {
        if Config::global().threads < 2 {
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
        if Config::global().threads < 2 {
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

    if Config::global().threads < 2 {
        (0..end).for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let l1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let l0 = l1 - dist;
            rx_apply_target(state_re, state_im, l0, l1, ct, nst);
        });
    } else {
        (0..end).into_par_iter().for_each(|i| {
            let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
            let l1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
            let l0 = l1 - dist;
            rx_apply_target(state_re, state_im, l0, l1, ct, nst);
        });
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

    if Config::global().threads < 2 {
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

    if Config::global().threads < 2 {
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

fn rz_apply_strategy1(state: &mut State, target: usize, diag_matrix: &[Amplitude; 2]) {
    let chunk_size = 1 << target;

    if Config::global().threads < 2 {
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

fn z_apply(state: &mut State, target: usize) {
    // NOTE: chunks == end >> target, where end == state.len() >> 1
    let chunks = state.len() >> (target + 1);

    if Config::global().threads < 2 {
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

    if Config::global().threads < 2 {
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

    if Config::global().threads < 2 {
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
        let l0 = base + i;
        let l1 = l0 + dist;
        u_apply_target(state_re, state_im, g, l0, l1);
    }
}

fn u_apply(state: &mut State, target: usize, theta: Float, phi: Float, lambda: Float) {
    let (st, ct) = Float::sin_cos(theta * 0.5);
    let (sl, cl) = Float::sin_cos(lambda);
    let (spl, cpl) = Float::sin_cos(phi + lambda);
    let cp = Float::cos(phi);

    let c = Amplitude { re: ct, im: 0.0 };
    let ncs = Amplitude {
        re: -cl * st,
        im: -sl * st,
    };
    let es = Amplitude {
        re: cp * st,
        im: sl * st,
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

    if Config::global().threads < 2 {
        (0..chunks).for_each(|chunk| {
            u_apply_strategy2(state_re, state_im, &g, chunk, target);
        });
    } else {
        (0..chunks).into_par_iter().for_each(|chunk| {
            u_apply_strategy2(state_re, state_im, &g, chunk, target);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::iqft,
        math::{pow2f, PI},
        utils::assert_float_closeness,
    };

    fn qcbm_functional(n: usize) -> State {
        let mut state = State::new(n);
        let pairs: Vec<_> = (0..n).map(|i| (i, (i + 1) % n)).collect();

        for i in 0..n {
            apply(Gate::RX(1.0), &mut state, i);
            apply(Gate::RZ(1.0), &mut state, i);
        }

        for i in 0..n - 1 {
            let (p0, p1) = pairs[i];
            c_apply(Gate::X, &mut state, p0, p1);
        }

        for _ in 0..9 {
            for i in 0..n {
                apply(Gate::RZ(1.0), &mut state, i);
                apply(Gate::RX(1.0), &mut state, i);
                apply(Gate::RZ(1.0), &mut state, i);
            }

            for i in 0..n - 1 {
                let (p0, p1) = pairs[i];
                c_apply(Gate::X, &mut state, p0, p1);
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
            apply(Gate::U((1.0, 1.0, 1.0)), &mut state, i);
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
}
