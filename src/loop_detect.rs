use crate::visualize::Visualizer;
use rustfft::num_complex::Complex;
use rustfft::Fft;
use rustfft::FftPlanner;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::OnceLock;

/// Indicates `end` (index of sample) repeats `start`.
pub struct Loop {
    /// Inclusive. Index of samples.
    pub start: usize,
    /// Exclusive. Index of samples.
    pub end: usize,
    /// Rough confidence of the loop.
    pub confidence: usize,
    /// For debugging purpose. Related to `end - start` but not the same.
    pub(crate) delta: usize,
}

/// States for loop detection.
pub struct LoopDetector {
    fft: OnceFft,
    fine_fft: OnceFft,
    chunk_size: usize,
    chunk_size_bits: u8,
    pub vis: Option<Visualizer>,
}

impl LoopDetector {
    pub fn new() -> Self {
        Self::new_with_chunk_size(11)
    }

    pub fn new_with_chunk_size(chunk_size_bits: u8) -> Self {
        let chunk_size = 1 << chunk_size_bits;
        // About 46ms per chunk.
        let fft = OnceFft::new(chunk_size_bits);
        // About 0.7ms per chunk. Used to align the loop more precisely.
        let fine_fft = OnceFft::new(5);
        Self {
            fft,
            fine_fft,
            chunk_size,
            chunk_size_bits,
            vis: None,
        }
    }

    pub fn enable_visualizer(mut self) -> Self {
        self.vis = Some(Visualizer::new());
        self
    }

    /// Find potential loops. The first loop is the most likely one and is fine tuned.
    /// `samples` should be Mono. Use `maybe_downmix` to convert Stereo to Mono.
    pub fn find_loops(&mut self, samples: &[i16]) -> Vec<Loop> {
        // Use rustfft to compute the FFT of the samples.
        // For each FFT frame, generate some hashes. Consider the previous frame to make hash more unique.
        // Maintain a multiple map from the hashes to timestamps (FFT frames).
        // If a hash appears in the map, it might match a previous timestamp.
        // Maintain another map from the timestamp delta to (the count of matched hashes, the start timestamp).
        // The ones of the largest matched count are likely loops.

        // Truncate to CHUNK_SIZE.
        let chunk_len = samples.len() >> self.chunk_size_bits;
        let samples = &samples[..chunk_len << self.chunk_size_bits];

        // FFT.
        let mut fft_buffer: Vec<Complex<f32>> = samples
            .iter()
            .map(|i| Complex::new(*i as f32, 0.0f32))
            .collect();
        let fft = self.fft.get_fft();
        fft.process_with_scratch(&mut fft_buffer, &mut self.fft.get_scratch());

        // Figure out `hash_to_timestamp` and `delta_to_starts`.
        let mut hash_to_timestamp: HashMap<u64, Vec<usize>> = Default::default();
        let mut delta_to_starts: HashMap<usize, Vec<usize>> = Default::default();

        let min_time_delta = chunk_len / 2;
        let mut last_hot_bands = HotBands::default();
        for (i, chunk) in fft_buffer.chunks_exact(self.chunk_size).enumerate() {
            let hot_bands = find_hot_bands(chunk);
            for &last_band in last_hot_bands.as_slice() {
                for &band in hot_bands.as_slice() {
                    let full_hash: u64 = (last_band as u64) | ((band as u64).wrapping_shl(32));
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
            last_hot_bands = hot_bands;
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

        for c in &mut counts {
            if c.count < count_threshold {
                break;
            }
            let start = pick_start(c.starts) << self.chunk_size_bits;
            let end = start + (c.delta << self.chunk_size_bits);
            let confidence = c.count;
            let delta = c.delta;
            loops.push(Loop {
                start,
                end,
                confidence,
                delta,
            });
        }

        if let Some(lop) = loops.first_mut() {
            self.fine_tune(samples, lop);
        }

        if let Some(vis) = self.vis.as_mut() {
            vis.push_fft_data(&fft_buffer, self.chunk_size);
            vis.push_loops(&loops);
            vis.push_matches(&delta_to_starts);
        }

        loops
    }

    /// Slightly shift `lop.end` so it can better align with `start`.
    pub fn fine_tune(&mut self, samples: &[i16], lop: &mut Loop) {
        let fft = self.fine_fft.get_fft();
        let scratch = self.fine_fft.get_scratch();

        // [##########]---------------------|----------|----------|
        // ^start     ^start+compare_size   ^          ^end       ^
        //                                  end_left      end_right
        //                                       [##########]
        //                                       ^best match

        // FFT for the start chunk.
        let compare_size = self.chunk_size * 4;
        let fft_start_norm: Vec<f32> = {
            let sample_start = match samples.get(lop.start..lop.start + compare_size) {
                None => return,
                Some(v) => v,
            };
            let mut fft_start_buffer: Vec<Complex<f32>> = sample_start
                .iter()
                .map(|i| Complex::new(*i as f32, 0.0f32))
                .collect();
            fft.process_with_scratch(&mut fft_start_buffer, scratch);
            fft_start_buffer.into_iter().map(|v| v.norm()).collect()
        };

        // Brute force search in range.
        let end_left = lop.end.saturating_sub(self.chunk_size).max(lop.start);
        let end_right = end_left + self.chunk_size * 2;
        let end_search_range: Vec<Complex<f32>> =
            match samples.get(end_left..end_right + compare_size) {
                None => return,
                Some(v) => v.iter().map(|i| Complex::new(*i as f32, 0.0f32)).collect(),
            };
        let mut best_offset = lop.end - end_left;
        let mut best_value = f32::MAX;
        let search_start = 0;
        let search_end = end_right - end_left;
        let step = 2;
        let mut fft_end_buffer: Vec<Complex<f32>> = Vec::with_capacity(self.chunk_size);
        for i in (search_start..search_end).step_by(step) {
            fft_end_buffer.clear();
            let slice = match end_search_range.get(i..i + compare_size) {
                None => break,
                Some(v) => v,
            };
            fft_end_buffer.extend_from_slice(slice);
            fft.process_with_scratch(&mut fft_end_buffer, scratch);
            let mut value = 0.0f32;
            for (a, b) in fft_start_norm.iter().zip(fft_end_buffer.iter()) {
                let diff = (a - b.norm()).abs();
                value += diff;
            }
            if value < best_value {
                best_value = value;
                best_offset = i;
            }
        }

        let new_end = end_left + best_offset;
        lop.end = new_end;
    }
}

/// Find the indexes of the "hot points".
/// For a typical 44100 Hz audio, with 2048 chunk size, the first 1024 values are useful,
/// and each value represents about 22 (22050 / 1024) Hz range.
pub(crate) fn find_hot_bands(fft_buffer: &[Complex<f32>]) -> HotBands {
    // Find top 3 points and their indexes.
    let mut top = [(0, 0.0f32); 3];
    // Attempt to skip nearby points.
    let short_distance_band = fft_buffer.len() >> 9;
    // Skip (less interesting) low frequency noise. ">> 8" skips <86 Hz for 44k Hz audio.
    let lower_band = fft_buffer.len() >> 8;
    for i in lower_band..(fft_buffer.len() >> 1) {
        let v = fft_buffer[i].norm();
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
pub(crate) fn pick_start(starts: &mut [usize]) -> usize {
    assert!(!starts.is_empty());
    // Histogram:
    //
    //   #####  #  #    #######  ##
    //   #               ######   #
    //                      # #
    //                        #
    //                  ^
    //                  block start
    // Find the first "block":
    // - Width >= BLOCK_SIZE_THRESHOLD.
    // - Average value >= BLOCK_AVERAGE_THRESHOLD.
    // - Only have small (< GAP_THRESHOLD) gaps.
    let histogram: Vec<usize> = {
        starts.sort_unstable();
        let mut histogram: Vec<usize> = Vec::new();
        let mut count = 0;
        let mut last = 0;
        for start in starts.iter().copied() {
            if start > last {
                histogram.push(count);
                for _i in last + 1..start {
                    histogram.push(0); // gap
                }
                count = 1;
                last = start;
            } else {
                debug_assert_eq!(start, last);
                count += 1;
            }
        }
        if count > 0 {
            histogram.push(count);
        }
        histogram
    };

    let mut block_start = 0usize; // inclusive
    let mut block_end = 0usize; // inclusive
    let mut block_total = 0usize;
    const GAP_THRESHOLD: usize = 3; // number of empty slots + 1
    const BLOCK_SIZE_THRESHOLD: usize = 6; // number of continuous matches
    const BLOCK_AVERAGE_THRESHOLD: usize = 2;
    for (i, v) in histogram.iter().enumerate() {
        let v = *v;
        if i > block_end + GAP_THRESHOLD {
            // Gap. Start a new block.
            block_start = i;
            block_end = i;
            block_total = 1;
        } else if v > 0 {
            block_total += v;
            block_end = i;
            let block_size = block_end - block_start + 1;
            if block_size >= BLOCK_SIZE_THRESHOLD
                && block_total >= BLOCK_AVERAGE_THRESHOLD * block_size
            {
                // Seems good enough.
                return block_start + starts[0];
            }
        }
    }

    // Fallback
    starts[0]
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
            .field("confidence", &self.confidence)
            .finish()
    }
}

struct OnceFft {
    chunk_size_bits: u8,
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
}