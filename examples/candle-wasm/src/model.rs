use std::collections::HashMap;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, VarBuilder};
use candle_transformers::models::qwen3;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::decode::{reconcile_secrets, spans_from_tags, viterbi_decode, Span};

const WINDOW: usize = 512;
const OVERLAP: usize = 64;
const MAX_TOKENS: usize = 4096;

#[derive(Deserialize)]
struct Config {
    #[serde(flatten)]
    model: qwen3::Config,
    model_type: String,
    privacy_attention_mode: String,
    id2label: HashMap<String, String>,
}

#[derive(Serialize)]
pub struct Detection {
    pub entities: Vec<Span>,
    pub redacted: String,
    pub tokens: usize,
}

pub struct Detector {
    config: qwen3::Config,
    weights: VarBuilder<'static>,
    tokenizer: Tokenizer,
    names: Vec<String>,
}

impl Detector {
    pub fn new(tensors: HashMap<String, Tensor>, config_json: &str, tokenizer_json: &[u8]) -> Result<Self, String> {
        let mut config: Config = serde_json::from_str(config_json).map_err(|_| "Invalid model config")?;
        if config.model_type != "qwen3" || config.privacy_attention_mode != "causal" || config.id2label.len() != 41 {
            return Err("Expected the released causal Qwen3 classifier with 41 labels".into());
        }
        let names: Vec<String> = (0..41).map(|i| config.id2label.get(&i.to_string()).cloned().ok_or("Missing model label"))
            .collect::<Result<_, _>>()?;
        config.model.max_position_embeddings = WINDOW;
        let weights = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let tokenizer = Tokenizer::from_bytes(tokenizer_json).map_err(|e| e.to_string())?;
        Ok(Self { config: config.model, weights, tokenizer, names })
    }

    pub fn detect(&self, text: &str) -> Result<Detection, String> {
        let encoding = self.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets();
        if ids.len() != offsets.len() || ids.len() > MAX_TOKENS {
            return Err("Input exceeds the demo limit of 4,096 tokens or has invalid offsets".into());
        }
        if ids.is_empty() { return Ok(Detection { entities: Vec::new(), redacted: text.into(), tokens: 0 }); }
        let mut totals = vec![vec![0.0_f32; self.names.len()]; ids.len()];
        let mut counts = vec![0_u32; ids.len()];
        for start in (0..ids.len()).step_by(WINDOW - OVERLAP) {
            let stop = (start + WINDOW).min(ids.len());
            // Qwen3 keeps a KV cache. Build a fresh model per window so independent
            // requests and overlapping windows cannot inherit previous context.
            let mut backbone = qwen3::Model::new(&self.config, self.weights.clone()).map_err(|e| e.to_string())?;
            let head = linear(self.config.hidden_size, self.names.len(), self.weights.pp("score")).map_err(|e| e.to_string())?;
            let tokens = Tensor::new(&ids[start..stop], &Device::Cpu).and_then(|t| t.unsqueeze(0)).map_err(|e| e.to_string())?;
            let logits = backbone.forward(&tokens, 0).and_then(|t| head.forward(&t)).and_then(|t| t.squeeze(0))
                .and_then(|t| t.to_vec2::<f32>()).map_err(|e| e.to_string())?;
            for (index, row) in logits.iter().enumerate() {
                for (sum, value) in totals[start + index].iter_mut().zip(row) { *sum += value; }
                counts[start + index] += 1;
            }
            if stop == ids.len() { break; }
        }
        for (row, count) in totals.iter_mut().zip(counts) {
            for value in row { *value /= count as f32; }
        }
        let tags = viterbi_decode(&totals, &self.names)?;
        let probabilities: Vec<f64> = totals.iter().zip(&tags).map(|(row, &tag)| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let denominator: f64 = row.iter().map(|&x| ((x as f64) - max).exp()).sum();
            ((row[tag] as f64) - max).exp() / denominator
        }).collect();
        let entities = spans_from_tags(text, offsets, &tags, &self.names, &probabilities)?;
        let entities = reconcile_secrets(text, offsets, &totals, &self.names, entities)?;
        let byte_positions: Vec<usize> = text.char_indices().map(|(i, _)| i).chain(std::iter::once(text.len())).collect();
        let mut redacted = text.to_string();
        for span in entities.iter().rev() {
            redacted.replace_range(byte_positions[span.start]..byte_positions[span.end], &format!("[{}]", span.label));
        }
        Ok(Detection { entities, redacted, tokens: ids.len() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_detector() -> Detector {
        let config: qwen3::Config = serde_json::from_str(r#"{
            "vocab_size":2,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,
            "num_attention_heads":2,"head_dim":2,"attention_bias":false,"num_key_value_heads":1,
            "max_position_embeddings":512,"sliding_window":null,"max_window_layers":1,
            "tie_word_embeddings":true,"rope_theta":10000.0,"rms_norm_eps":0.000001,
            "use_sliding_window":false,"hidden_act":"silu"
        }"#).unwrap();
        let tokenizer = Tokenizer::from_bytes(br#"{
            "version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,
            "pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"hi":0,"[UNK]":1},"unk_token":"[UNK]"}
        }"#).unwrap();
        let mut names = vec!["O".to_string()];
        for category in ["PERSON", "EMAIL", "PHONE", "ADDRESS", "GOVERNMENT_ID", "FINANCIAL_ID", "SECRET", "IP_ADDRESS", "DATE_OF_BIRTH", "URL"] {
            for prefix in ['B', 'I', 'E', 'S'] { names.push(format!("{prefix}-{category}")); }
        }
        Detector { config, weights: VarBuilder::zeros(DType::F32, &Device::Cpu), tokenizer, names }
    }

    #[test]
    fn resets_state_between_overlapping_windows_and_independent_requests() {
        let detector = tiny_detector();
        let text = "hi ".repeat(513);
        let first = detector.detect(&text).unwrap();
        assert_eq!(first.tokens, 513);
        assert_eq!(first.redacted, text);
        assert!(first.entities.is_empty());
        let _ = detector.detect("hi hi").unwrap();
        let second = detector.detect(&text).unwrap();
        assert_eq!(serde_json::to_string(&first).unwrap(), serde_json::to_string(&second).unwrap());
    }
}
