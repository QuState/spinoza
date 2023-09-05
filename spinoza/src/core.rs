//! Abstractions for representing a Quantum State
use crate::{
    config::Config,
    gates::{apply, Gate},
    math::{modulus, pow2f, Float, PI},
};
use once_cell::sync::OnceCell;
use rand::prelude::*;
use std::collections::HashMap;

/// Reference to the Config for user passed config args
pub static CONFIG: OnceCell<Config> = OnceCell::new();

#[repr(C, align(32))]
#[derive(Copy, Clone)]
pub struct Wrappedf64x4([Float; 4]);

#[derive(Clone)]
pub struct State {
    pub reals: Vec<Wrappedf64x4>,
    pub imags: Vec<Wrappedf64x4>,
    pub n: u8,
}

impl State {
    /// Create a new State. The state will always be of size 2^{n},
    /// where n is the number of qubits. Note that n cannot be 0.
    pub fn new(n: usize) -> Self {
        assert!(n > 2);
        let mut reals = vec![Wrappedf64x4([0.0, 0.0, 0.0, 0.0]); (1 << n) >> 2];
        let imags = vec![Wrappedf64x4([0.0, 0.0, 0.0, 0.0]); (1 << n) >> 2];
        reals[0] = Wrappedf64x4([1.0, 0.0, 0.0, 0.0]);
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
