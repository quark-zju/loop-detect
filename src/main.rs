use miniserde::json;
use miniserde::Serialize;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::borrow::Cow;
use std::collections::BTreeMap;
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

struct Opts {
    play: bool,
    debug: bool,
    around: Option<f64>,
    quiet: bool,
    batch: bool,
}

#[derive(Serialize, Clone, Copy)]
struct HumanReadableLoop {
    // units are in "seconds", not samples
    start: f32,
    end: f32,
    confidence: f32,
}

macro_rules! vprintln {
    ($opts:ident, $($arg:tt)*) => {
        if !$opts.quiet {
            eprintln!($($arg)*);
        }
    };
}

fn cli() -> io::Result<()> {
    let mut pargs = pico_args::Arguments::from_env();
    if pargs.contains(["-h", "--help"]) {
        return help();
    }
    let play = pargs.contains(["-p", "--play"]);
    let debug = pargs.contains("--debug");
    let quiet = pargs.contains(["-q", "--quiet"]);
    let jobs: usize = match pargs.opt_value_from_fn(["-j", "--jobs"], |v| v.parse::<usize>()) {
        Ok(Some(v)) => v,
        Ok(None) => std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1),
        _ => 1,
    };
    let around = pargs
        .opt_value_from_fn(["-A", "--around"], |s| s.parse::<f64>())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let input_files = {
        let mut v = Vec::new();
        while let Ok(Some(input_file)) = pargs.opt_free_from_str::<String>() {
            v.push(input_file);
        }
        v
    };
    if input_files.is_empty() {
        eprintln!("No input files.");
        return help();
    }

    let batch = input_files.len() > 1;
    let opts = Opts {
        play,
        debug,
        around,
        quiet,
        batch,
    };

    let mut json_str = None;
    if !batch {
        if let Some(input_file) = input_files.first() {
            if let Some(lop) = process_one(input_file, &opts)? {
                json_str = Some(json::to_string(&lop));
            }
        }
    } else if jobs <= 1 {
        let mut reports: BTreeMap<&str, Option<HumanReadableLoop>> = BTreeMap::new();
        for input_file in input_files.iter() {
            match process_one(input_file, &opts) {
                Ok(lop) => {
                    reports.insert(input_file, lop);
                }
                Err(e) => {
                    reports.insert(input_file, None);
                    vprintln!(opts, "{}: {}", input_file, e);
                }
            }
        }
        json_str = Some(json::to_string(&reports));
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(jobs.min(input_files.len()))
            .build()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        pool.install(|| {
            let reports: BTreeMap<&String, Option<HumanReadableLoop>> = input_files
                .as_slice()
                .into_par_iter()
                .map(|input_file| match process_one(input_file, &opts) {
                    Ok(lop) => (input_file, lop),
                    Err(e) => {
                        vprintln!(opts, "{}: {}", input_file, e);
                        (input_file, None)
                    }
                })
                .collect();
            json_str = Some(json::to_string(&reports));
        })
    }

    if let Some(json_str) = json_str {
        println!("{}", json_str);
    }

    Ok(())
}

const MINIMAL_CONFIDENCE: f32 = 0.2;

fn process_one(input_file: &str, opts: &Opts) -> io::Result<Option<HumanReadableLoop>> {
    let mut info = ProbeInfo::default();
    if opts.play {
        // Loop analysis only needs 1 channel. Only read both channels with --play.
        info = ffmpeg::probe(&input_file).unwrap_or(info);
    }

    vprintln!(
        opts,
        "Decoding ({} Hz, {} channels)...",
        &info.sample_rate,
        info.channels
    );
    let samples = ffmpeg::decode(&input_file, info)?;

    vprintln!(opts, "Finding loops...");
    let mut detector = loop_detect::LoopDetector::new();
    if opts.debug {
        detector = detector.enable_visualizer();
    }
    let loops = {
        let mono_samples = maybe_downmix(&samples, info.channels);
        detector.find_loops(&mono_samples)
    };

    if let Some(vis) = detector.vis.as_ref() {
        let html = vis.export_html();
        let file_name = format!("{}.html", &input_file);
        vprintln!(opts, "Writing debug HTML: {}", &file_name);
        fs::write(&file_name, html)?;
    }

    match loops.first() {
        Some(lop) if lop.confidence > MINIMAL_CONFIDENCE => {
            let human_loop = HumanReadableLoop {
                start: lop.start as f32 / info.sample_rate as f32,
                end: lop.end as f32 / info.sample_rate as f32,
                confidence: lop.confidence,
            };
            if opts.play {
                vprintln!(
                    opts,
                    "Playing with loop [{} to {} secs, confidence: {}].",
                    human_loop.start,
                    human_loop.end,
                    human_loop.confidence
                );
                let play_result = match opts.around {
                    None => ffmpeg::play_loop(&samples, lop, info),
                    Some(around) => ffmpeg::play_loop_around(&samples, lop, info, around),
                };
                if !opts.batch {
                    play_result?;
                }
            }
            Ok(Some(human_loop))
        }
        _ => {
            vprintln!(opts, "No loop found");
            Ok(None)
        }
    }
}

fn help() -> io::Result<()> {
    println!(
        "Usage: [options] <audio file> [audio file]...

Options:
    -h, --help         Print this help message.
    -p, --play         Play the looped audio. Requires `ffplay`.
    -A, --around SECS  Only play SECS before loop end, and SECS after loop start.
        --debug        Generate a HTML file for debugging.
    -q, --quiet        Silence verbose (stderr) output.
    -j, --jobs         CPU cores to use for processing multiple files.

Prints the loop information {{\"start\": seconds, \"end\": seconds, \"confidence\": zero_to_one}}.
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
