//! Configuration options for runnning spinoza
use crate::core::CONFIG;
use clap::Parser;
use num_cpus;

/// Config for simulations that are run using the CLI
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// The number of threads to distribute the worload amongst.
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
            threads: num_cpus::get().try_into().unwrap(),
            qubits: 25,
            print: false,
        }
    }
}

/// Representation of the CLI args
#[derive(Parser, Debug)]
pub struct QSArgs {
    #[clap(short, long)]
    threads: u32,
    #[clap(short, long)]
    print: bool,
    #[clap(short, long)]
    qubits: u8,
}
