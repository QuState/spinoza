//! Abstractions for a quantum logic gates
use crate::{
    config::Config,
    core::{State, Wrappedf64x4},
    math::{Amplitude, Float, SQRT_ONE_HALF},
};
use rayon::prelude::*;
use std::simd::*;

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
        Gate::RZ(theta) => rz_apply(state, target, theta),
        _ => (),
    }
}

unsafe fn cast_slice_mut<A: Copy, B: Copy>(a: &mut [A]) -> &mut [B] {
    let new_len = core::mem::size_of_val(a) / core::mem::size_of::<B>();
    unsafe { core::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut B, new_len) }
}

fn rz_apply_strategy1(state: &mut State, target: usize, diag_matrix: &[Amplitude; 2]) {
    let chunk_size = 1 << target;
    let (reals, imags) = unsafe {
        let reals = cast_slice_mut::<Wrappedf64x4, Float>(&mut state.reals);
        let imags = cast_slice_mut::<Wrappedf64x4, Float>(&mut state.imags);
        (reals, imags)
    };

    if chunk_size >= 4 {
        reals
            .chunks_exact_mut(chunk_size)
            .zip(imags.chunks_exact_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (c0, c1))| {
                let m_re = f64x4::splat(diag_matrix[i & 1].re);
                let m_im = f64x4::splat(diag_matrix[i & 1].im);

                for (a, b) in c0
                    .as_simd_mut::<4>()
                    .1
                    .iter_mut()
                    .zip(c1.as_simd_mut::<4>().1.iter_mut())
                {
                    let v0 = *a * m_re - *b * m_im;
                    let v1 = *a * m_im + *b * m_re;
                    *a = v0;
                    *b = v1;
                }
            });
    } else {
        reals
            .chunks_exact_mut(chunk_size)
            .zip(imags.chunks_exact_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (c0, c1))| {
                let m_re = f64x2::splat(diag_matrix[i & 1].re);
                let m_im = f64x2::splat(diag_matrix[i & 1].im);
                for (a, b) in c0
                    .as_simd_mut::<2>()
                    .1
                    .iter_mut()
                    .zip(c1.as_simd_mut::<2>().1.iter_mut())
                {
                    let v0 = *a * m_re - *b * m_im;
                    let v1 = *a * m_im + *b * m_re;
                    *a = v0;
                    *b = v1;
                }
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
