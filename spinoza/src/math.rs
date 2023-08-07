//! An assortment of mathematical structures, functions, and constants for quantum state
//! simulation.

/// An alias for <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_1_SQRT_2.html>
pub const SQRT_ONE_HALF: Float = std::f64::consts::FRAC_1_SQRT_2 as Float;

/// An alias for <https://doc.rust-lang.org/std/f64/consts/constant.PI.html>
pub const PI: Float = std::f64::consts::PI as Float;

/// The type of floating point number to use for amplitudes
#[cfg(feature = "double")]
pub type Float = f64;

#[cfg(feature = "single")]
pub type Float = f32;

/// An amplitude that makes up a Quantum State
#[derive(Copy, Clone)]
pub struct Amplitude {
    /// imaginary component
    pub im: Float,
    /// real component
    pub re: Float,
}

/// The absolute value of a complex number
/// See <https://en.wikipedia.org/wiki/Absolute_value#Complex_numbers>
#[inline]
pub fn modulus(z_re: Float, z_im: Float) -> Float {
    (z_re.mul_add(z_re, z_im * z_im)).sqrt()
}

/// Compute 2^n and convert it to a float
pub const fn pow2f(n: usize) -> Float {
    (1 << n) as Float
}
