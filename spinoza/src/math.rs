pub const SQRT_ONE_HALF: Float = std::f64::consts::FRAC_1_SQRT_2 as Float;
pub const PI: Float = std::f64::consts::PI as Float;

#[cfg(feature = "double")]
pub type Float = f64;

#[cfg(feature = "single")]
pub type Float = f32;

#[derive(Copy, Clone)]
pub struct Amplitude {
    pub im: Float,
    pub re: Float,
}

#[inline]
pub fn modulus(z_re: Float, z_im: Float) -> Float {
    (z_re.mul_add(z_re, z_im * z_im)).sqrt()
}

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

pub const fn log_2(x: usize) -> usize {
    (num_bits::<usize>() as u32 - x.leading_zeros() - 1) as usize
}

pub const fn pow2f(n: usize) -> Float {
    (1 << n) as Float
}

pub const fn pow2u(n: usize) -> usize {
    1 << n
}
