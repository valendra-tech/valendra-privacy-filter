use std::collections::HashMap;

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Entry {
    pub name: String,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

struct Pending {
    entry: Entry,
    values: Vec<f32>,
    bytes_received: u64,
}

pub struct StreamingWeights {
    entries: Vec<Entry>,
    next: usize,
    pending: Option<Pending>,
    tensors: HashMap<String, Tensor>,
    hash: Sha256,
    expected_hash: String,
}

impl StreamingWeights {
    pub fn new(prefix: &[u8], expected_hash: &str) -> Result<Self, String> {
        if prefix.len() < 8 || expected_hash.len() != 64 || !expected_hash.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("Invalid Safetensors header or checksum".into());
        }
        let length = u64::from_le_bytes(prefix[..8].try_into().map_err(|_| "Invalid header")?);
        if length > 64 * 1024 * 1024 || length as usize != prefix.len() - 8 {
            return Err("Invalid Safetensors header length".into());
        }
        let header: serde_json::Value = serde_json::from_slice(&prefix[8..]).map_err(|_| "Invalid Safetensors JSON")?;
        let mut entries = Vec::new();
        for (name, value) in header.as_object().ok_or("Invalid Safetensors index")? {
            if name == "__metadata__" { continue; }
            if value["dtype"].as_str() != Some("F32") { return Err("Expected FP32 weights".into()); }
            let shape: Vec<usize> = value["shape"].as_array().ok_or("Invalid tensor shape")?.iter()
                .map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()).ok_or("Invalid tensor shape"))
                .collect::<Result<_, _>>()?;
            let offsets = value["data_offsets"].as_array().ok_or("Invalid tensor offsets")?;
            if offsets.len() != 2 { return Err("Invalid tensor offsets".into()); }
            let start = offsets[0].as_u64().ok_or("Invalid tensor offsets")?;
            let end = offsets[1].as_u64().ok_or("Invalid tensor offsets")?;
            let elements = shape.iter().try_fold(1usize, |acc, &n| acc.checked_mul(n)).ok_or("Tensor too large")?;
            if end < start || elements.checked_mul(4).map(|n| n as u64) != Some(end - start) {
                return Err("Tensor length does not match shape".into());
            }
            entries.push(Entry { name: name.clone(), shape, start, end });
        }
        entries.sort_by_key(|entry| entry.start);
        let mut offset = 0;
        for entry in &entries {
            if entry.start != offset { return Err("Tensor ranges overlap or have gaps".into()); }
            offset = entry.end;
        }
        if entries.is_empty() { return Err("No tensors found".into()); }
        let mut hash = Sha256::new();
        hash.update(prefix);
        Ok(Self { entries, next: 0, pending: None, tensors: HashMap::new(), hash, expected_hash: expected_hash.to_ascii_lowercase() })
    }

    pub fn entries(&self) -> &[Entry] { &self.entries }

    pub fn begin(&mut self, name: &str) -> Result<(), String> {
        if self.pending.is_some() { return Err("A tensor is already in progress".into()); }
        let entry = self.entries.get(self.next).ok_or("All tensors have already been loaded")?;
        if entry.name != name { return Err("Unexpected tensor order".into()); }
        let elements = usize::try_from((entry.end - entry.start) / 4).map_err(|_| "Tensor too large")?;
        let mut values = Vec::new();
        values.try_reserve_exact(elements).map_err(|_| "Insufficient WASM memory for model weights")?;
        self.pending = Some(Pending { entry: entry.clone(), values, bytes_received: 0 });
        Ok(())
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<(), String> {
        let pending = self.pending.as_mut().ok_or("No tensor in progress")?;
        if bytes.len() % 4 != 0 || pending.bytes_received + bytes.len() as u64 > pending.entry.end - pending.entry.start {
            return Err("Invalid tensor chunk length".into());
        }
        self.hash.update(bytes);
        pending.values.extend(bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
        pending.bytes_received += bytes.len() as u64;
        Ok(())
    }

    pub fn end(&mut self) -> Result<(), String> {
        let pending = self.pending.as_ref().ok_or("No tensor in progress")?;
        if pending.bytes_received != pending.entry.end - pending.entry.start { return Err("Incomplete tensor".into()); }
        let pending = self.pending.take().unwrap();
        let tensor = Tensor::from_vec(pending.values, pending.entry.shape.as_slice(), &Device::Cpu).map_err(|e| e.to_string())?;
        self.tensors.insert(pending.entry.name, tensor);
        self.next += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<HashMap<String, Tensor>, String> {
        if self.pending.is_some() || self.next != self.entries.len() { return Err("Incomplete model weights".into()); }
        if format!("{:x}", self.hash.finalize()) != self.expected_hash { return Err("Model SHA-256 mismatch".into()); }
        Ok(self.tensors)
    }
}
