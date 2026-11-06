# Privacy Filter: Candle/WebAssembly demo

This experimental demo runs the published `valendra/privacy-filter-mini` checkpoint entirely in the browser using Rust, Candle and WebAssembly SIMD. The static server serves the HTML and compiled module only; the input text is not sent to an inference API.

**Live demo:** <https://valendra-tech.github.io/valendra-privacy-filter/>

The model revision is pinned to `62a73ef2a5a216b0e0d0d5d2aac4e869847f3983`. The demo downloads the original 2.38 GB FP32 Safetensors file from Hugging Face using bounded HTTP ranges and verifies its SHA-256 before inference. Config and tokenizer hashes are also checked. No quantized or alternative weights are used. A browser may require substantially more than 2.38 GB of available memory, and a reload downloads the weights again.

## Build and run

Install Rust with the `wasm32-unknown-unknown` target, Node.js, and Python 3. Then, from this directory:

```bash
rustup target add wasm32-unknown-unknown
RUSTFLAGS="-C target-feature=+simd128" npx --yes wasm-pack@0.15.1 build --target web --release --out-dir web/pkg
python3 -m http.server 8099 --bind 127.0.0.1 --directory web
```

Open <http://127.0.0.1:8099/>, click **Cargar modelo**, wait for the verified download and enter text. The browser must support WebAssembly SIMD and cross-origin HTTP range requests. Run the static server on localhost or HTTPS so the browser provides the cryptographic API. If multiple Rust toolchains are installed, ensure the `rustc` used by `wasm-pack` is the one with the WASM target installed.

The detector uses 512-token windows with a 64-token overlap, averages overlapping logits, performs constrained BIOES decoding and applies the secret-fragment reconciliation rules. Independent windows and requests use fresh Qwen3 state. It refuses inputs over 4,096 tokens instead of truncating them. Scores are not calibrated. This browser port is experimental; tests with a small set of inputs do not establish full numerical parity or real-world privacy protection.

## Verification

```bash
cargo test --tests
npm test
```

The real-checkpoint test is explicitly ignored unless requested. To compare inference and repeated-request behavior with a locally available copy of the released checkpoint:

```bash
PRIVACY_FILTER_CHECKPOINT=/path/to/checkpoint-4000 \
  cargo test --release --test real_checkpoint -- --ignored
```

The tokenizer offset check can likewise be run with `PRIVACY_FILTER_TOKENIZER=/path/to/tokenizer.json cargo test --test tokenizer_offsets -- --ignored`.
