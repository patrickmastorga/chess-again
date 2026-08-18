use std::env;
use std::fs;
use std::path::Path;

// simple pseudorandom number generator
pub struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        // increment the state
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15u64);

        // apply MurmurHash3 mixer to state
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9u64);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111ebu64);
        z ^ (z >> 31)
    }
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("zobrist.rs");

    let mut pieces = [0u64; 12 * 64];
    let mut en_passant = [0u64; 8];
    let mut castling = [0u64; 4];

    let mut rng = SplitMix64::new(0x9E3779B97F4A7C15u64);
    for slot in pieces.iter_mut() {
        *slot = rng.next_u64();
    }
    for slot in en_passant.iter_mut() {
        *slot = rng.next_u64();
    }
    for slot in castling.iter_mut() {
        *slot = rng.next_u64();
    }
    let color = rng.next_u64();

    // emit a source file that defines a static instance of the Zobrist struct.
    // `include!`d into the `zobrist` module, so it can access the private fields
    let mut out = String::new();
    out.push_str("pub static ZOBRIST: Zobrist = Zobrist {\n");
    out.push_str("    pieces: [\n");
    for v in pieces.iter() {
        out.push_str(&format!("        0x{:016x},\n", v));
    }
    out.push_str("    ],\n");
    out.push_str("    en_passant: [\n");
    for v in en_passant.iter() {
        out.push_str(&format!("        0x{:016x},\n", v));
    }
    out.push_str("    ],\n");
    out.push_str("    castling: [\n");
    for v in castling.iter() {
        out.push_str(&format!("        0x{:016x},\n", v));
    }
    out.push_str("    ],\n");
    out.push_str(&format!("    color: 0x{:016x},\n", color));
    out.push_str("};\n");

    fs::write(&dest, out).unwrap();
}
