use std::borrow::Cow;
use std::fs;
use std::io;

pub(crate) mod ffmpeg;
pub(crate) mod loop_detect;
pub(crate) mod visualize;

use crate::ffmpeg::ProbeInfo;

fn main() {
    if let Err(e) = cli() {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

fn cli() -> io::Result<()> {
    let mut pargs = pico_args::Arguments::from_env();
    if pargs.contains(["-h", "--help"]) {
        return help();
    }
    let play = pargs.contains(["-p", "--play"]);
    let debug = pargs.contains("--debug");
    let around = pargs
        .opt_value_from_fn(["-A", "--around"], |s| s.parse::<f64>())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let input_file = pargs
        .free_from_str::<String>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Missing input file."))?;

    let mut info = ProbeInfo::default();
    if play {
        // Loop analysis only needs 1 channel. Only read both channels with --play.
        info = ffmpeg::probe(&input_file).unwrap_or(info);
    }

    eprintln!(
        "Decoding ({} Hz, {} channels)...",
        &info.sample_rate, info.channels
    );
    let samples = ffmpeg::decode(&input_file, info)?;

    eprintln!("Finding loops...");
    let mut detector = loop_detect::LoopDetector::new();
    if debug {
        detector = detector.enable_visualizer();
    }
    let loops = {
        let mono_samples = maybe_downmix(&samples, info.channels);
        detector.find_loops(&mono_samples)
    };

    if let Some(vis) = detector.vis.as_ref() {
        let html = vis.export_html();
        let file_name = format!("{}.html", input_file);
        eprintln!("Writing debug HTML: {}", &file_name);
        fs::write(&file_name, html)?;
    }

    if let Some(lop) = loops.first() {
        println!(
            "{{\"start\": {}, \"end\": {}, \"confidence\": {}}}",
            lop.start as f64 / info.sample_rate as f64,
            lop.end as f64 / info.sample_rate as f64,
            lop.confidence
        );
        if play {
            eprintln!("Playing...");
            match around {
                None => ffmpeg::play_loop(&samples, lop, info)?,
                Some(around) => ffmpeg::play_loop_around(&samples, lop, info, around)?,
            }
        }
    } else {
        eprintln!("No loop found");
    }

    Ok(())
}

fn help() -> io::Result<()> {
    println!(
        "Usage: [options] <audio file>

Options:
    -h, --help         Print this help message.
    -p, --play         Play the looped audio. Requires `ffplay`.
    -A, --around SECS  Only play SECS before loop end, and SECS after loop start.
        --debug        Generate a HTML file for debugging.

Prints the loop information {{\"start\": seconds, \"end\": seconds}}.
Requires `ffmpeg` to decode `<audio file>`."
    );
    return Ok(());
}

/// Downmix to Mono for easier processing.
fn maybe_downmix(samples: &[i16], channels: usize) -> Cow<[i16]> {
    if channels <= 1 {
        Cow::Borrowed(samples)
    } else {
        let mixed: Vec<i16> = samples
            .chunks_exact(channels)
            .map(|chunk| chunk.iter().copied().sum::<i16>() / channels as i16)
            .collect();
        Cow::Owned(mixed)
    }
}
