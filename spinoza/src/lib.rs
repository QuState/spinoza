#![warn(clippy::complexity)]
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
pub mod utils;
