//! The rust core of Spinoza
#![warn(clippy::complexity)]
#![warn(missing_docs)]
#![warn(clippy::style)]
#![warn(clippy::correctness)]
#![warn(clippy::suspicious)]
#![warn(clippy::perf)]
#![deny(unsafe_op_in_unsafe_fn)]
#![feature(slice_swap_unchecked)]

pub mod circuit;
pub mod config;
pub mod core;
pub mod gates;
pub mod math;
pub mod measurement;
pub mod unitaries;
pub mod utils;
