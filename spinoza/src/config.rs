use crate::core::CONFIG;
use clap::Parser;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub print: bool,
    pub qubits: u8,
}

impl Config {
    pub fn global() -> &'static Config {
        CONFIG.get().expect("config is not initialized")
    }

    pub const fn from_cli(args: QSArgs) -> Config {
        Config {
            qubits: args.qubits,
            print: args.print,
        }
    }

    pub const fn benchmark() -> Config {
        Config {
            qubits: 25,
            print: false,
        }
    }

    pub const fn test() -> Config {
        Config {
            qubits: 25,
            print: false,
        }
    }
}

#[derive(Parser, Debug)]
pub struct QSArgs {
    #[clap(short, long)]
    print: bool,
    #[clap(short, long)]
    qubits: u8,
}
