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
pub fn pow2f(n: usize) -> Float {
    const BASE2: Float = 2.0;
    BASE2.powi(n.try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::assert_float_closeness;

    fn linspace(start: Float, end: Float, num: Option<usize>) -> Vec<Float> {
        let n = if let Some(num) = num { num } else { 50 };
        let step = (end - start) / n as Float;
        let mut x = 0.0;
        let mut res: Vec<Float> = Vec::with_capacity(n);

        while x < end {
            res.push(x);
            x += step;
        }
        res
    }

    #[test]
    fn modulus_unit_circle() {
        let angles = linspace(0.0, 2.0 * PI, Some(100));
        for angle in angles.into_iter() {
            let amplitude = Amplitude {
                re: angle.cos(),
                im: angle.sin(),
            };
            assert_float_closeness(modulus(amplitude.re, amplitude.im), 1.0, 0.001);
        }
    }

    #[test]
    fn pow() {
        for i in 0..52 {
            let mut res = 1.0;
            for _ in 0..i {
                res *= 2.0;
            }
            assert_float_closeness(pow2f(i), res, 0.001);
        }
    }
}
