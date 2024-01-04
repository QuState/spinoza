//! Configuration options for running spinoza
use clap::Parser;

use crate::core::CONFIG;

/// Config for simulations that are run using the CLI
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// The number of threads to distribute the workload.
    /// `u32` is used to represent number of threads since 4,294,967,295 is a
    /// reasonable upperbound. If you have access to a matrioshka brain, and you
    /// need a larger data type, please reach out.
    pub threads: u32,
    /// Whether or not to print the State represented as a table.
    pub print: bool,
    /// The number of qubits that will make up the State.  State vector size is 2^{n}, where n is
    /// the # of qubits. Assuming single precision complex numbers, the upper bound with u8 is
    /// 2^255 * 64 bit ≈ 4.632 * 10^{65} TB (terabytes). Thus, using u8 suffices.
    pub qubits: u8,
}

impl Config {
    /// Get or init the global Config. The default
    pub fn global() -> &'static Config {
        CONFIG.get_or_init(Config::test)
    }

    /// Convert the provided CLI args and turn it into a Config
    pub const fn from_cli(args: QSArgs) -> Config {
        assert!(args.threads > 0 && args.qubits > 0);
        Config {
            threads: args.threads,
            qubits: args.qubits,
            print: args.print,
        }
    }

    fn test() -> Config {
        Config {
            threads: std::thread::available_parallelism()
                .unwrap()
                .get()
                .try_into()
                .expect("Too much power"),
            // no input for tests, so this quantity should not matter
            qubits: 0,
            print: false,
        }
    }
}

/// Representation of the CLI args
#[derive(Parser)]
#[command(author, version, about)]
pub struct QSArgs {
    /// Number of threads to use
    #[clap(short, long)]
    threads: u32,
    /// Whether or not to print the state in tabular format
    #[clap(short, long)]
    print: bool,
    /// The number of qubits to use in the system
    #[clap(short, long)]
    qubits: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_args() {
        let config = Config::global();
        assert_eq!(
            config.threads,
            u32::try_from(std::thread::available_parallelism().unwrap().get()).unwrap()
        );
        assert_eq!(config.qubits, 0);
        assert!(!config.print);
    }
}
