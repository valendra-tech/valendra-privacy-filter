use sha2::{Digest, Sha256};
use valendra_candle_wasm::weights::StreamingWeights;

fn fixture() -> (Vec<u8>, Vec<u8>, String) {
    let header = br#"{"example":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
    prefix.extend_from_slice(header);
    let data = [1.0_f32.to_le_bytes(), 2.0_f32.to_le_bytes()].concat();
    let mut digest = Sha256::new();
    digest.update(&prefix);
    digest.update(&data);
    (prefix, data, format!("{:x}", digest.finalize()))
}

#[test]
fn loads_and_verifies_a_tensor_incrementally() {
    let (prefix, data, hash) = fixture();
    let mut loader = StreamingWeights::new(&prefix, &hash).unwrap();
    assert_eq!(loader.entries()[0].name, "example");
    loader.begin("example").unwrap();
    loader.push(&data[..4]).unwrap();
    loader.push(&data[4..]).unwrap();
    loader.end().unwrap();
    let tensors = loader.finish().unwrap();
    assert_eq!(tensors["example"].to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn rejects_truncated_corrupted_and_out_of_order_data() {
    let (prefix, data, hash) = fixture();
    assert!(StreamingWeights::new(&prefix[..prefix.len()-1], &hash).is_err());
    let mut loader = StreamingWeights::new(&prefix, &hash).unwrap();
    assert!(loader.begin("wrong").is_err());
    loader.begin("example").unwrap();
    loader.push(&data[..4]).unwrap();
    assert!(loader.end().is_err());

    let mut loader = StreamingWeights::new(&prefix, &hash).unwrap();
    loader.begin("example").unwrap();
    loader.push(&[0; 8]).unwrap();
    loader.end().unwrap();
    assert!(loader.finish().is_err());
}
