use std::collections::HashMap;

use crate::loop_detect::find_hot_bands;
use crate::loop_detect::Loop;
use rustfft::num_complex::Complex;

pub struct Visualizer {
    body: String,
    js: String,
}

static HEAD_JS: &str = include_str!("visualize.js");
static TAIL_JS: &str =
    "window.addEventListener('load', () => render(data, hotPoints, loops, matches));";
static STYLE: &str = include_str!("visualize.css");

impl Visualizer {
    pub fn new() -> Self {
        Self {
            body: String::new(),
            js: String::new(),
        }
    }

    pub fn export_html(&self) -> String {
        return format!(
            "<html><head><meta charset=\"UTF-8\"><style>{}</style><script>{}{}{}</script></head><body>{}</body></html>",
            STYLE, HEAD_JS, self.js, TAIL_JS, self.body,
        );
    }

    pub(crate) fn push_fft_data(&mut self, data: &[Complex<f32>], chunk_size: usize) {
        self.push_js("const data = [\n");
        let mut hot_points_str = String::new();
        for chunk in data.chunks_exact(chunk_size) {
            let hot_points = find_hot_bands(chunk);
            hot_points_str.push_str(&format!("{:?},", hot_points.as_slice()));
            let f32_array = chunk[..chunk_size / 2]
                .iter()
                .map(|v| v.norm())
                .collect::<Vec<_>>();
            self.push_js(&format!("  new Float32Array({:?}),", f32_array));
        }
        self.push_js("];\n");
        self.push_js(&format!("const hotPoints = [{}];\n", hot_points_str));

        let width = data.len() / chunk_size;
        let height = chunk_size / 2;
        self.push_html(&format!(
            "<canvas id=\"canvas\" width=\"{}\" height=\"{}\"></canvas>",
            width, height
        ));
    }

    pub(crate) fn push_loops(&mut self, loops: &[Loop]) {
        self.push_js("const loops = [\n");
        for l in loops {
            self.push_js(&format!(
                "  {{start: {}, end: {}, confidence: {}, delta: {}}},\n",
                l.start, l.end, l.confidence, l.delta
            ));
        }
        self.push_js("];\n");
    }

    pub(crate) fn push_matches(&mut self, delta_to_starts: &HashMap<usize, Vec<usize>>) {
        self.push_js("const matches = new Map([\n");
        for (delta, starts) in delta_to_starts {
            self.push_js(&format!("  [{}, {:?}],\n", delta, starts,));
        }
        self.push_js("]);\n");
    }

    pub(crate) fn push_fine_tune(
        &mut self,
        left: &[f32],
        right: &[f32],
        chunk_size: usize,
        delta: usize,
        confidence: f32,
    ) {
        assert_eq!(left.len(), right.len());
        self.push_js(&format!(
            "fineTunes.push({{a: new Float32Array({:?}), b: new Float32Array({:?}), chunkSize: {}, delta: {}, confidence: {}}});\n",
            left, right, chunk_size, delta, confidence
        ));
    }

    fn push_js(&mut self, js: &str) {
        self.js.push_str(js);
    }

    fn push_html(&mut self, html: &str) {
        self.body.push_str(html);
    }
}
