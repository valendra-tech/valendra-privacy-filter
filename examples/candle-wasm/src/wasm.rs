use wasm_bindgen::prelude::*;

use crate::model::Detector;
use crate::weights::StreamingWeights;

const WEIGHTS_SHA256: &str = "130ea7b3a1de096aa5a1c32500d82a8157546c83607d5f71e361df0710fb9d7b";

fn error(message: impl ToString) -> JsError { JsError::new(&message.to_string()) }

#[wasm_bindgen]
pub struct ModelLoader { weights: Option<StreamingWeights> }

#[wasm_bindgen]
impl ModelLoader {
    #[wasm_bindgen(constructor)]
    pub fn new(prefix: &[u8]) -> Result<ModelLoader, JsError> {
        Ok(Self { weights: Some(StreamingWeights::new(prefix, WEIGHTS_SHA256).map_err(error)?) })
    }

    pub fn plan(&self) -> Result<String, JsError> {
        serde_json::to_string(self.weights.as_ref().ok_or_else(|| error("Loader already consumed"))?.entries()).map_err(error)
    }

    pub fn begin_tensor(&mut self, name: &str) -> Result<(), JsError> {
        self.weights.as_mut().ok_or_else(|| error("Loader already consumed"))?.begin(name).map_err(error)
    }

    pub fn push_chunk(&mut self, bytes: &[u8]) -> Result<(), JsError> {
        self.weights.as_mut().ok_or_else(|| error("Loader already consumed"))?.push(bytes).map_err(error)
    }

    pub fn end_tensor(&mut self) -> Result<(), JsError> {
        self.weights.as_mut().ok_or_else(|| error("Loader already consumed"))?.end().map_err(error)
    }

    pub fn finish(&mut self, config: &str, tokenizer: &[u8]) -> Result<BrowserDetector, JsError> {
        let weights = self.weights.take().ok_or_else(|| error("Loader already consumed"))?.finish().map_err(error)?;
        Ok(BrowserDetector { detector: Detector::new(weights, config, tokenizer).map_err(error)? })
    }
}

#[wasm_bindgen]
pub struct BrowserDetector { detector: Detector }

#[wasm_bindgen]
impl BrowserDetector {
    pub fn detect(&self, text: &str) -> Result<String, JsError> {
        serde_json::to_string(&self.detector.detect(text).map_err(error)?).map_err(error)
    }
}
