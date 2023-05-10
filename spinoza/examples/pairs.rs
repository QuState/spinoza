use spinoza::utils::padded_bin;

fn print_pairs() {
    let timer = std::time::Instant::now();
    let n = 6;
    let p2n = 1 << n;

    let t = 0;
    let p2t = 1 << t;

    let c = 3;
    let p2c = 1 << (c + 1);

    let p2m = 1 << (c - t - 1);

    for i in 0..p2m {
        for j in (2 * i + 2 * p2m + 1) * p2t..(2 * i + 2 * p2m + 2) * p2t {
            // println!("{} = {}", j, padded_bin(j, n));
            for k in (j..j + p2n).step_by(p2c) {
                println!("{} = {}", k, padded_bin(k, n));
            }
        }
    }

    println!("time elapsed: {} us ... ", timer.elapsed().as_micros());
}

fn print_target_pairs() {
    let n = 3;

    let target = 1;
    let dist = 1 << target;

    // 2 for loops
    println!("for loops");
    for i in 0..1 << (n - 1 - target) {
        for j in 2 * i * dist..(2 * i + 1) * dist {
            println!(
                "{} = {} -> {} = {}",
                j,
                padded_bin(j, n),
                dist + j,
                padded_bin(dist + j, n)
            )
        }
    }

    // divmod
    println!("\ndivmod");
    for i in 0..1 << (n - 1) {
        // let (p, s) = (i / dist, i % dist);
        // i = dist*p + s
        let j = i + ((i >> target) << target); //i + (p << target); // i + dist*p; // 2 * dist * p + s;
        println!(
            "{} = {} -> {} = {}",
            j,
            padded_bin(j, n),
            dist + j, // i + ((1 + (i >> target)) << target),
            padded_bin(dist + j, n)
        )
    }

    // bit manipulation
    println!("\nbit manipulation");
    let neg_dist = !0 << target;
    for i in 0..1 << (n - 1) {
        let j = i + (i & neg_dist);
        println!(
            "{} = {} -> {} = {}",
            j,
            padded_bin(j, n),
            dist + j,
            padded_bin(dist + j, n)
        )
    }

    for target in 0..n {
        for i in 0..1 << (n - 1) {
            assert_eq!(
                (i >> target) << target,
                i & !0 << target,
                "different for {}",
                i
            );
        }
    }
}

fn main() {
    print_target_pairs();
    print_pairs();
}
