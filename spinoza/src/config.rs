use crate::core::CONFIG;
use clap::Parser;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub threads: u32, // If a larger data type is needed, that's just wild
    pub print: bool,
    pub qubits: u8, // State vector size is 2^{n}, where n is the # of qubits. Assuming single
                    // precision complex numbers, the upper bound with u8 is 2^255 * 64 bit ≈
                    // 4.632 * 10^{65} TB (terabytes). Thus, using u8 suffices.
}

impl Config {
    pub fn global() -> &'static Config {
        CONFIG.get_or_init(Config::test)
    }

    pub const fn from_cli(args: QSArgs) -> Config {
        assert!(args.threads > 0 && args.qubits > 0);
        Config {
            threads: args.threads,
            qubits: args.qubits,
            print: args.print,
        }
    }

    pub const fn benchmark() -> Config {
        Config {
            threads: 1,
            qubits: 25,
            print: false,
        }
    }

    pub const fn test() -> Config {
        Config {
            threads: 14,
            qubits: 25,
            print: false,
        }
    }
}

#[derive(Parser, Debug)]
pub struct QSArgs {
    #[clap(short, long)]
    threads: u32,
    #[clap(short, long)]
    print: bool,
    #[clap(short, long)]
    qubits: u8,
}
