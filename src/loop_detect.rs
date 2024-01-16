use crate::visualize::Visualizer;
use rustfft::num_complex::Complex;
use rustfft::Fft;
use rustfft::FftPlanner;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::sync::OnceLock;

/// Indicates `end` (index of sample) repeats `start`.
pub struct Loop {
    /// Inclusive. Index of samples.
    pub start: usize,
    /// Exclusive. Index of samples.
    pub end: usize,
    /// The confidence of the "start" (on the "rough" FFT).
    pub start_confidence: f32,
    /// Rough confidence of the loop (on the "fine" FFT).
    pub end_confidence: f32,
    /// For debugging purpose. Related to `end - start` but not the same.
    pub(crate) delta: usize,
    /// For debugging purpose. Hash matches.
    pub(crate) votes: usize,
}

/// States for loop detection.
pub struct LoopDetector {
    fft: OnceFft,
    fine_fft: OnceFft,
    pub vis: Option<Visualizer>,
}

impl LoopDetector {
    pub fn new() -> Self {
        Self::new_with_chunk_size(11)
    }

    pub fn new_with_chunk_size(chunk_size_bits: u8) -> Self {
        // About 46ms per chunk ((1 << 11) / 44.100).
        let fft = OnceFft::new(chunk_size_bits);
        // About 0.7ms per chunk. Used to align the loop more precisely.
        let fine_fft = OnceFft::new(5);
        Self {
            fft,
            fine_fft,
            vis: None,
        }
    }

    pub fn enable_visualizer(mut self) -> Self {
        self.vis = Some(Visualizer::new());
        self
    }

    pub fn find_loops(&mut self, samples: &[i16]) -> Vec<Loop> {
        find_potential_loops(&mut self.fft, &mut self.fine_fft, samples, &mut self.vis)
    }
}

/// Find potential loops. The first loop is the most likely one and is fine tuned.
/// `samples` should be Mono. Use `maybe_downmix` to convert Stereo to Mono.
fn find_potential_loops(
    fft: &mut OnceFft,
    fine_fft: &mut OnceFft,
    samples: &[i16],
    vis: &mut Option<Visualizer>,
) -> Vec<Loop> {
    // Use rustfft to compute the FFT of the samples.
    // For each FFT frame, generate some hashes. Consider the previous frame to make hash more unique.
    // Maintain a multiple map from the hashes to timestamps (FFT frames).
    // If a hash appears in the map, it might match a previous timestamp.
    // Maintain another map from the timestamp delta to (the count of matched hashes, the start timestamp).
    // The ones of the largest matched count are likely loops.

    // Truncate to CHUNK_SIZE.
    let chunk_size = fft.chunk_size();
    let chunk_size_bits = fft.chunk_size_bits;
    let chunk_len = samples.len() >> chunk_size_bits;
    let samples = &samples[..chunk_len << chunk_size_bits];

    // FFT.
    let mut fft_buffer: Vec<Complex<f32>> = samples
        .iter()
        .map(|i| Complex::new(*i as f32, 0.0f32))
        .collect();
    let afft = fft.get_fft();
    afft.process_with_scratch(&mut fft_buffer, &mut fft.get_scratch());
    let fft_buffer = normalize_complex_chunks(&fft_buffer, chunk_size, false);

    // Figure out `hash_to_timestamp` and `delta_to_starts`.
    let mut hash_to_timestamp: HashMap<u64, Vec<usize>> = Default::default();
    let mut delta_to_starts: HashMap<usize, Vec<usize>> = Default::default();

    let min_time_delta = chunk_len / 2;
    const LAST_FRAME_BITS: usize = 2;
    const LAST_FRAME_COUNT: usize = 1 << LAST_FRAME_BITS;
    let mut last_hot_bands_deq: VecDeque<HotBands> =
        (0..LAST_FRAME_COUNT).map(|_| HotBands::default()).collect();
    for (i, chunk) in fft_buffer.chunks_exact(chunk_size).enumerate() {
        let hot_bands = find_hot_bands(chunk);
        for (j, last_hot_bands) in last_hot_bands_deq.iter().enumerate() {
            for &last_band in last_hot_bands.as_slice() {
                for &band in hot_bands.as_slice() {
                    if last_band.abs_diff(band) <= 2 {
                        // Skip similar band -> band changes.
                        continue;
                    }
                    let full_hash: u64 = (((last_band as u64) | ((band as u64).wrapping_shl(30)))
                        << LAST_FRAME_BITS)
                        | (j as u64);
                    match hash_to_timestamp.entry(full_hash) {
                        Entry::Occupied(mut e) => {
                            // Avoid O(N^2) by limiting the count of previous samples.
                            const MAX_PREVIOUS: usize = 60;
                            for &previous_timestamp in e.get().iter().take(MAX_PREVIOUS) {
                                let delta = i - previous_timestamp;
                                // Exclude too short loops.
                                if delta > min_time_delta {
                                    delta_to_starts
                                        .entry(delta)
                                        .or_default()
                                        .push(previous_timestamp);
                                }
                            }
                            e.get_mut().push(i);
                        }
                        Entry::Vacant(e) => {
                            e.insert(vec![i]);
                        }
                    }
                }
            }
        }
        last_hot_bands_deq.pop_front();
        last_hot_bands_deq.push_back(hot_bands);
    }

    // Find the max count in `delta_to_starts`.
    #[derive(Debug)]
    struct Count<'a> {
        delta: usize,
        count: usize,
        starts: &'a mut [usize],
    }
    let mut counts: Vec<Count> = delta_to_starts
        .iter_mut()
        .map(|(delta, starts)| {
            let delta = *delta;
            let count = starts.len();
            Count {
                delta,
                count,
                starts,
            }
        })
        .collect();
    counts.sort_unstable_by_key(|v| -(v.count as isize));
    counts.truncate(5);

    let mut loops = Vec::new();
    let count_threshold = match counts.first() {
        None => return loops,
        Some(v) => v.count / 2,
    };

    let mut best_confidence = 0f32;
    let mut best_index = 0;
    for c in &mut counts {
        if c.count < count_threshold {
            break;
        }
        if loops.iter().any(|l| l.delta.abs_diff(c.delta) < 10) {
            // Skip "similar" loops.
            continue;
        }
        dbg!(c.count, c.delta);
        let (start, start_confidence) = pick_start(c.starts, &fft_buffer, c.delta, chunk_size_bits);
        let start = start << chunk_size_bits;
        let end = start + (c.delta << chunk_size_bits);
        let (end, end_confidence) = fine_tune(fft.chunk_size(), fine_fft, samples, start, end, vis);
        let delta = c.delta;
        let overall_confidence = end_confidence * 0.7 + start_confidence * 0.3;
        if overall_confidence > best_confidence {
            best_confidence = overall_confidence;
            best_index = loops.len();
        }
        loops.push(Loop {
            start,
            end,
            start_confidence,
            end_confidence,
            delta,
            votes: c.count,
        });
        if end_confidence > 0.9 {
            // Good enough. Do not try other loops.
            break;
        }
    }

    if best_index > 0 {
        // Move the best loop to the front.
        loops.swap(0, best_index);
    }

    if let Some(vis) = vis {
        vis.push_fft_data(&fft_buffer, chunk_size);
        vis.push_matches(&delta_to_starts);
        vis.push_loops(&loops);
    }

    loops
}

/// Slightly shift `end` so it can better align with `start`.
/// Both `start` and `end` are in FFT frame.
/// Returns `(end, confidence)`.
fn fine_tune(
    rough_chunk_size: usize,
    fine_fft: &mut OnceFft,
    samples: &[i16],
    start: usize,
    end: usize,
    vis: &mut Option<Visualizer>,
) -> (usize, f32) {
    let afft = fine_fft.get_fft();
    let chunk_size_bits = fine_fft.chunk_size_bits;
    let chunk_size = fine_fft.chunk_size();
    let scratch = fine_fft.get_scratch();

    // [##########]---------------------|----------|----------|
    // ^start     ^start+compare_size   ^          ^end       ^
    //                                  end_left      end_right
    //                                       [##########]
    //                                       ^best match

    // FFT for the start chunk.
    let compare_chunk_count = rough_chunk_size >> chunk_size_bits;
    if compare_chunk_count < 3 {
        return (end, 0.0);
    }
    let compare_size = chunk_size * compare_chunk_count;
    let fft_start_norm: Vec<f32> = {
        let sample_start = match samples.get(start..start + compare_size) {
            None => return (end, 0.0),
            Some(v) => v,
        };
        let mut fft_start_buffer: Vec<Complex<f32>> = sample_start
            .iter()
            .map(|i| Complex::new(*i as f32, 0.0f32))
            .collect();
        afft.process_with_scratch(&mut fft_start_buffer, scratch);
        normalize_complex_chunks(&fft_start_buffer, chunk_size, true)
    };

    // Brute force search in range.
    let end_left = end.saturating_sub(compare_size).max(start);
    let end_right = end + compare_size;
    let end_search_range: Vec<Complex<f32>> = match samples.get(end_left..end_right + compare_size)
    {
        None => return (end, 0.0),
        Some(v) => v.iter().map(|i| Complex::new(*i as f32, 0.0f32)).collect(),
    };
    let mut best_offset = end - end_left;
    let mut best_value = -1f32;
    let search_start = 0;
    let search_end = end_right - end_left;
    let step = 1;
    let mut fft_end_buffer: Vec<Complex<f32>> = Vec::with_capacity(compare_size);
    let mut buf: Vec<f32> = vec![0f32; compare_size];
    for i in (search_start..search_end).step_by(step) {
        fft_end_buffer.clear();
        let slice = match end_search_range.get(i..i + compare_size) {
            None => break,
            Some(v) => v,
        };
        fft_end_buffer.extend_from_slice(slice);
        afft.process_with_scratch(&mut fft_end_buffer, scratch);

        normalize_complex_chunks_in_place(&fft_end_buffer, &mut buf, chunk_size, true);
        let value = calculate_similarity(&fft_start_norm, &buf);
        assert!(value >= 0.0 && value <= 1.0);
        if value > best_value {
            best_value = value;
            best_offset = i;
        }
    }

    if let Some(vis) = vis.as_mut() {
        fft_end_buffer.clear();
        fft_end_buffer
            .extend_from_slice(&end_search_range[best_offset..best_offset + compare_size]);
        afft.process_with_scratch(&mut fft_end_buffer, scratch);
        let buf = normalize_complex_chunks(&fft_end_buffer, chunk_size, true);
        vis.push_fine_tune(&fft_start_norm, &buf, chunk_size, best_value);
    }

    let new_end = end_left + best_offset;
    (new_end, best_value)
}

fn normalize_f32_chunks_in_place(buf: &mut [f32], chunk_size: usize, normalize_volume: bool) {
    // normalize each chunk so volumn (ex. fade out) affects comparsion less.
    let effective_size = chunk_size / 2;
    let low_freq = (effective_size >> 8).max(1);
    let high_freq = effective_size - (effective_size >> 2);
    let band = 22050 / chunk_size;
    for chunk in buf.chunks_exact_mut(chunk_size) {
        // zero low-freq noise.
        // zero high-freq potential noise (ex. MP3 artifacts).
        // scale by log(freq).
        for (i, v) in chunk.iter_mut().enumerate() {
            if i >= high_freq || i < low_freq {
                *v = 0.0;
            } else {
                *v *= (((i + 1) * band) as f32).log2();
            }
        }
        if normalize_volume {
            // Find max.
            let mut max: f32 = 0.0;
            for &v in chunk.iter().take(effective_size) {
                if v > max {
                    max = v;
                }
            }
            if max < 1e-3 {
                continue;
            }
            // Attempt to scale to max = 100, with a log-scale volumn.
            let scale = (100.0f32 + max.log10() * 2.0) / max;
            for v in chunk.iter_mut().take(effective_size) {
                *v *= scale;
            }
        }
    }
}

fn normalize_complex_chunks_in_place(
    buf: &[Complex<f32>],
    out_buf: &mut [f32],
    chunk_size: usize,
    normalize_volume: bool,
) {
    assert_eq!(buf.len(), out_buf.len());
    // complex -> f32
    for i in 0..buf.len() {
        out_buf[i] = buf[i].norm();
    }
    normalize_f32_chunks_in_place(out_buf, chunk_size, normalize_volume);
}

fn normalize_complex_chunks(
    buf: &[Complex<f32>],
    chunk_size: usize,
    normalize_volume: bool,
) -> Vec<f32> {
    // complex -> f32
    let mut buf: Vec<f32> = buf.into_iter().map(|v| v.norm()).collect();
    normalize_f32_chunks_in_place(&mut buf, chunk_size, normalize_volume);
    buf
}

/// Find the indexes of the "hot points".
/// For a typical 44100 Hz audio, with 2048 chunk size, the first 1024 values are useful,
/// and each value represents about 22 (22050 / 1024) Hz range.
pub(crate) fn find_hot_bands(fft_buffer: &[f32]) -> HotBands {
    // Find top 3 points and their indexes.
    let mut top = [(0, 0.0f32); 3];
    // Attempt to skip nearby points.
    let short_distance_band = fft_buffer.len() >> 9;
    // Skip (less interesting) low frequency noise. ">> 8" skips <86 Hz for 44k Hz audio.
    let lower_band = fft_buffer.len() >> 8;
    for i in lower_band..(fft_buffer.len() >> 1) {
        let v = fft_buffer[i];
        if v > top[0].1 {
            if i >= top[0].0 + short_distance_band {
                top[2] = top[1];
                top[1] = top[0];
            }
            top[0] = (i, v);
        } else if i >= top[0].0 + short_distance_band {
            if v > top[1].1 {
                if i >= top[1].0 + short_distance_band {
                    top[2] = top[1];
                }
                top[1] = (i, v);
            } else if v > top[2].1 && i > top[1].0 + short_distance_band {
                top[2] = (i, v);
            }
        }
    }
    let threshold = top[0].1 / 4.0;
    let mut out = HotBands::default();
    for (i, v) in top {
        if v > threshold {
            out.push(i as _);
        }
    }
    out
}

/// Pick a good "start". The "start" can shift within
/// a matching range:
///
/// ```plain,ignore
///     start       end (valid (start, end) pair)
///     v           v
/// +-----------------+
/// |   /           / |
/// |  /           /  |
/// | #           %   |
/// +-----------------+
///   ^ start     ^ end (another valid (start, end) pair)
///
/// Note if  the "#" does not match "%" closely then "#" is not a good start.
/// ```
///
/// Example track that needs this:
/// Rainbowdragoneyes/(2018) The Messenger OST - Disc II- The Future/26 The Corrupted Future
pub(crate) fn pick_start(
    starts: &mut [usize],
    fft_buffer: &[f32],
    delta_frame: usize,
    chunk_size_bits: u8,
) -> (usize, f32) {
    assert!(!starts.is_empty());

    // Find the "best" by sliding and comparing N frames.
    // By default N = 8, chunk_size_bits = 11, for 44k Hz audio,
    // the N frames cover: (N << chunk_size_bits) / 44100 = 0.37s.
    const N: usize = 8;

    starts.sort_unstable();
    let mut rolling_index = 0;
    let mut rolling_values = [0f32; N];
    let mut rolling_total = 0f32;
    let mut last_start = 0;
    let mut best_value = 0f32;
    let mut best_offset = starts.first().copied().unwrap_or_default();
    for &mut start in starts {
        if start <= last_start {
            // Handled before.
            continue;
        }

        if start >= last_start + 3 {
            // Gap is too large. Reset rolling values.
            rolling_values = Default::default();
            rolling_total = 0.0;
            rolling_index = 0;
        }

        let end = last_start + delta_frame;
        // [ ---- ]           [ ---- ]
        //        ^ start     ^ end (last_start + delta)
        // ^ last_start
        let start_range = (last_start << chunk_size_bits)..(start << chunk_size_bits);
        let end_range = (end << chunk_size_bits)..((end + start - last_start) << chunk_size_bits);
        let a = match fft_buffer.get(start_range) {
            None => break,
            Some(v) => v,
        };
        let b = match fft_buffer.get(end_range) {
            None => break,
            Some(v) => v,
        };

        let value = calculate_similarity(a, b);
        let old_value = rolling_values[rolling_index];
        rolling_values[rolling_index] = value;
        rolling_total += value - old_value;
        rolling_index += 1;
        if rolling_index >= N {
            rolling_index = 0;
        }

        if rolling_total > best_value {
            best_value = rolling_total;
            best_offset = last_start.saturating_sub(N - 1);
        }

        last_start = start;
    }

    (best_offset, best_value / N as f32)
}

fn calculate_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut total_diff = 0.0f32;
    let mut total_power = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        let diff = (a - b).abs();
        total_diff += diff;
        total_power += a.max(b);
    }
    1.0 - (total_diff / total_power)
}

#[derive(Copy, Clone, Default)]
pub(crate) struct HotBands {
    pub len: usize,
    pub data: [u32; 3],
}

impl HotBands {
    pub fn push(&mut self, v: u32) {
        assert!(self.len < 3);
        self.data[self.len] = v;
        self.len += 1;
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.data[..self.len]
    }
}

impl fmt::Debug for Loop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Loop")
            .field("start", &self.start)
            .field("len", &(self.end - self.start))
            .field("confidence", &self.end_confidence)
            .finish()
    }
}

struct OnceFft {
    pub(crate) chunk_size_bits: u8,
    fft: OnceLock<Arc<dyn Fft<f32>>>,
    scratch: Vec<Complex<f32>>,
}

impl OnceFft {
    fn new(chunk_size_bits: u8) -> Self {
        Self {
            chunk_size_bits,
            fft: OnceLock::new(),
            scratch: Vec::new(),
        }
    }

    fn get_fft(&self) -> Arc<dyn Fft<f32>> {
        self.fft
            .get_or_init(|| {
                let chunk_size = 1 << self.chunk_size_bits;
                let mut planner = FftPlanner::new();
                planner.plan_fft_forward(chunk_size)
            })
            .clone()
    }

    fn get_scratch(&mut self) -> &mut [Complex<f32>] {
        if self.scratch.is_empty() {
            let fft = self.get_fft();
            self.scratch
                .resize(fft.get_inplace_scratch_len(), Complex::default());
        }
        &mut self.scratch
    }

    fn chunk_size(&self) -> usize {
        1 << self.chunk_size_bits
    }
}
