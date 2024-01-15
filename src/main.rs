#[cfg(not(feature = "cli"))]
compile_error!("Enable the `cli` feature to build the CLI.");

pub(crate) mod ffmpeg;

#[cfg(feature = "cli")]
pub(crate) mod cli;

fn main() {
    #[cfg(feature = "cli")]
    if let Err(e) = cli::cli() {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
