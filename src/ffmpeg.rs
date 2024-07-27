use loop_detect::Loop;
use std::io;
use std::io::Read;
use std::io::Write;
use std::process::Command;
use std::process::Stdio;

/// Decode an audio file to a vector of samples via the `ffmpeg` executable.
pub fn decode(path: &str, info: ProbeInfo) -> io::Result<Vec<i16>> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-i",
            path,
            "-v",
            "warning",
            "-vn",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
        ])
        .arg(info.channels.to_string())
        .arg("-ar")
        .arg(info.sample_rate.to_string())
        .arg("-")
        .stdout(Stdio::piped())
        .spawn()?;

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Failed to capture ffmpeg stdout"))?;

    let mut buffer = Vec::with_capacity(1 << 25);
    stdout.read_to_end(&mut buffer)?;

    let exit_status = child.wait()?;
    if !exit_status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "ffmpeg exited with non-zero status: {:?}",
                exit_status.code()
            ),
        ));
    }

    let samples = buffer
        .chunks_exact(2)
        .map(|chunk| chunk[0] as i16 | ((chunk[1] as i16) << 8))
        .collect();

    Ok(samples)
}

/// Return value of `probe`.
#[derive(Copy, Clone, Debug)]
pub struct ProbeInfo {
    pub sample_rate: usize,
    pub channels: usize,
}

const DEFAULT_SAMPLE_RATE: usize = 44100;

impl Default for ProbeInfo {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            channels: 1,
        }
    }
}

impl ProbeInfo {
    fn sample_count_seconds(&self, seconds: f64) -> usize {
        ((self.sample_rate as f64 * seconds) as usize) * self.channels
    }

    fn channel_layout(&self) -> &str {
        match self.channels {
            1 => "mono",
            2 => "stereo",
            v => panic!("unable to convert channel count {} to layout", v),
        }
    }
}

/// Find the sample rate via the `ffprobe` executable.
pub fn probe(path: &str) -> io::Result<ProbeInfo> {
    let out = Command::new("ffprobe")
        .args(["-v", "quiet", "-show_streams", path])
        .output()?;

    let mut info = ProbeInfo::default();
    if let Ok(stdout) = std::str::from_utf8(&out.stdout) {
        for line in stdout.lines() {
            if let Some(v) = line.strip_prefix("sample_rate=") {
                if let Ok(v) = v.trim_end().parse::<usize>() {
                    info.sample_rate = v;
                }
            } else if let Some(v) = line.strip_prefix("channels=") {
                if let Ok(v) = v.trim_end().parse::<usize>() {
                    info.channels = v;
                }
            }
        }
    }

    Ok(info)
}

/// Play via the `ffplay` executable.
pub fn play(samples: impl IntoIterator<Item = i16>, info: ProbeInfo) -> io::Result<()> {
    let mut child = Command::new("ffplay")
        .args([
            "-loglevel",
            "warning",
            "-vn",
            "-nodisp",
            "-f",
            "s16le",
            "-i",
            "-",
            "-ch_layout",
        ])
        .arg(info.channel_layout())
        .arg("-ar")
        .arg(info.sample_rate.to_string())
        .stdin(Stdio::piped())
        .spawn()?;

    let mut stdin = match child.stdin.take() {
        Some(stdin) => stdin,
        None => {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Failed to capture ffplay stdin",
            ))
        }
    };
    let mut buf = [0u8; 1 << 14];
    let mut i = 0;
    for sample in samples {
        let bytes = sample.to_le_bytes();
        let next = i + bytes.len();
        buf[i..next].copy_from_slice(&bytes);
        i = next;
        if i >= buf.len() {
            stdin.write_all(&buf[..])?;
            i = 0;
        }
    }
    stdin.write_all(&buf[..i])?;
    drop(stdin);

    Ok(())
}

/// Play `samples[..end]`, then repeat `samples[start..end]`.
pub fn play_loop(samples: &[i16], lop: &Loop, info: ProbeInfo) -> io::Result<()> {
    let channels = info.channels;
    let iter = samples[..lop.end * channels].iter().copied();
    let iter = iter.chain(
        samples[lop.start * channels..lop.end * channels]
            .iter()
            .copied()
            .cycle(),
    );
    play(iter, info)
}

/// Play in a loop:
/// - `samples[(end - around_samples)..end]`
/// - `samples[start..(start + around_samples)]`
/// - silence for 1 second
pub fn play_loop_around(
    samples: &[i16],
    lop: &Loop,
    info: ProbeInfo,
    around: f64,
) -> io::Result<()> {
    let channels = info.channels;
    let end = lop.end * channels;
    let around = info.sample_count_seconds(around);
    let mut chunk = samples[end.saturating_sub(around)..end].to_vec();
    let start = lop.start * channels;
    chunk.extend_from_slice(&samples[start..((start + around).min(end))]);
    chunk.resize(chunk.len() + info.sample_count_seconds(1.0), 0);
    let iter = chunk.into_iter().cycle();
    play(iter, info)
}
