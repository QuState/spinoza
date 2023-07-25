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

// TODO(Saveliy): optimize for fun
/// Convert an unsigned 128 bit integer to a more legible String.
pub fn pretty_print_int(i: u128) -> String {
    let mut s = String::new();
    let i_str = i.to_string();
    let a = i_str.chars().rev().enumerate();
    for (idx, val) in a {
        if idx != 0 && idx % 3 == 0 {
            s.insert(0, ',');
        }
        s.insert(0, val);
    }
    s
}

/// Convert a `usize` to its binary expansion, but padded with 0's. Padding is of size, width.
pub fn padded_bin(i: usize, width: usize) -> String {
    format!("{:01$b}", i, width + 2)
}

/// Display a the `State` as a table
pub fn to_table(state: &State) {
    let n: usize = state.n.into();
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        // .set_content_arrangement(ContentArrangement::Dynamic)
        .set_width(100)
        .set_header(vec![
            Cell::new("Outcome"),
            Cell::new("Amplitude"),
            Cell::new("Magnitude"),
            Cell::new("Amplitude Bar"),
            Cell::new("Probability"),
            Cell::new("Probability Bar"),
        ]);

    (0..16.min(state.len())).for_each(|idx| {
        let z_re = state.reals[idx];
        let z_im = state.imags[idx];
        table.add_row(vec![
            Cell::new(format!("{} = {}", idx, padded_bin(idx, n))),
            Cell::new(format!("{:.5} + i{:.5}", z_re, z_im)),
            Cell::new(format!("{:.5}", modulus(z_re, z_im))),
            Cell::new(str::repeat(
                " ",
                (modulus(z_re, z_im) * 50.0).round() as usize,
            ))
            .bg(complex_to_rgb(z_re, z_im, false)),
            Cell::new(format!("{:.5}", modulus(z_re, z_im).powi(2))),
            Cell::new(str::repeat(
                " ",
                (modulus(z_re, z_im).powi(2) * 50.0).round() as usize,
            ))
            .bg(complex_to_rgb(1.0, 0.0, false)),
        ]);
    });
    table.force_no_tty().enforce_styling().style_text_only();
    println!("{}", table);
}

fn complex_to_rgb(z_re: Float, z_im: Float, scaled_saturation: bool) -> Color {
    let val = 100.0;

    let mut hue: f32 = (z_im.atan2(z_re) * 180.0 / PI) as f32;
    if hue < 0.0 {
        hue += 360.0;
    }

    let sat: f32 = if scaled_saturation {
        modulus(z_re, z_im) as f32 * 100.0
    } else {
        100.0
    };

    let [r, g, b] = hsv_to_rgb(hue, sat, val);
    Rgb { r, g, b }
}

// https://gist.github.com/eyecatchup/9536706 Colors
fn hsv_to_rgb(hue: f32, sat: f32, val: f32) -> [u8; 3] {
    // Make sure our arguments stay in-range
    let (mut h, mut s, mut v) = (
        0.0_f32.max(360.0_f32.min(hue)),
        0.0_f32.max(100.0_f32.min(sat)),
        0.0_f32.max(100.0_f32.min(val)),
    );

    // We accept saturation and value arguments from 0 to 100 because that's
    // how Photoshop represents those values. Internally, however, the
    // saturation and value are calculated from a range of 0 to 1.
    // We make that conversion here.
    s /= 100.0;
    v /= 100.0;

    let (r, g, b) = (
        (v * 255.0).round(),
        (v * 255.0).round(),
        (v * 255.0).round(),
    );

    if s == 0.0 {
        // Achromatic(grey)
        return [r as u8, g as u8, b as u8];
    }

    h /= 60.0; // sector 0 to 5
    let i = h.floor() as i16;
    let f = h - f32::from(i); // factorial part of h
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let (r, g, b) = if i == 0 {
        (v, t, p)
    } else if i == 1 {
        (q, v, p)
    } else if i == 2 {
        (p, v, t)
    } else if i == 3 {
        (p, q, v)
    } else if i == 4 {
        (t, p, v)
    } else {
        (v, p, q)
    };
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    ]
}

/// Create an iterator of `Range`'s, such that the total size is
/// `total_count`, and the ranges are of approximately equal size. Deprecated.
pub fn balanced_ranges(
    total_count: usize,
    bucket_count: usize,
) -> impl Iterator<Item = std::ops::Range<usize>> {
    let b = bucket_count.min(total_count);
    let (q, r) = (total_count / b, total_count % b);
    let mut start: usize = 0;

    (0..b).map(move |i| {
        let range = std::ops::Range {
            start,
            end: start + q + if i < r { 1 } else { 0 },
        };
        start = range.end;
        range
    })
}

/// Asserts that two floating point numbers are approximately equal.
pub fn assert_float_closeness(actual: Float, expected: Float, epsilon: Float) {
    assert!((actual - expected).abs() < epsilon);
}
