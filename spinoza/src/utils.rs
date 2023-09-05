//! An assortment of utility functions for visualizing, benchmarking, and testing.
use crate::{
    core::State,
    math::{modulus, Float, PI},
};
use comfy_table::{
    presets::UTF8_FULL,
    Color::Rgb,
    {Cell, Color, Table},
};

/// Formats an unsigned, 128 bit integer with commas, as a string. Used for readability
pub fn pretty_print_int(i: u128) -> String {
    if i == 0 {
        return "0".into();
    }

    // u128::MAX == 340_282_366_920_938_463_463_374_607_431_768_211_455
    // len(340_282_366_920_938_463_463_374_607_431_768_211_455") == 51
    let mut q = arrayvec::ArrayVec::<u8, 51>::new();

    let mut x = i;
    let mut comma = 0;

    while x > 0 {
        let r = x % 10;
        x /= 10;

        if comma == 3 {
            q.push(44); // 44 is ',' in ASCII
            comma = 0;
        }
        q.push((0x30 + r) as u8); // ascii digits 0, 1, 2, ... start at value 0x30
        comma += 1;
    }

    q.into_iter().map(|d| d as char).rev().collect()
}

/// Convert a `usize` to its binary expansion, but padded with 0's. Padding is of size, width.
pub fn padded_bin(i: usize, width: usize) -> String {
    format!("{:01$b}", i, width + 2)
}

/// Asserts that two floating point numbers are approximately equal.
pub fn assert_float_closeness(actual: Float, expected: Float, epsilon: Float) {
    assert!((actual - expected).abs() < epsilon);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pretty_print_int() {
        for i in 0..1000 {
            assert_eq!(pretty_print_int(i), i.to_string());
        }

        assert_eq!(pretty_print_int(1_000), "1,000");
        assert_eq!(pretty_print_int(10_000), "10,000");
        assert_eq!(pretty_print_int(100_000), "100,000");
        assert_eq!(pretty_print_int(1_000_000), "1,000,000");
        assert_eq!(pretty_print_int(1_000_000_000), "1,000,000,000");
        assert_eq!(pretty_print_int(1_000_000_000_000), "1,000,000,000,000");
        assert_eq!(pretty_print_int(100_000_000_000_000), "100,000,000,000,000");
        assert_eq!(
            pretty_print_int(u128::MAX),
            "340,282,366,920,938,463,463,374,607,431,768,211,455"
        );
    }
}
