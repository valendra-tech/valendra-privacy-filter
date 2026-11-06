//! Candle-backed browser inference for Valendra Privacy Filter.

pub mod decode;
pub mod model;
pub mod weights;

#[cfg(target_arch = "wasm32")]
mod wasm;
