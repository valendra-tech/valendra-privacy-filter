use std::io::Read;
use std::path::Path;

use valendra_candle_wasm::model::Detector;
use valendra_candle_wasm::weights::StreamingWeights;

#[test]
#[ignore = "requires the released checkpoint via PRIVACY_FILTER_CHECKPOINT"]
fn real_checkpoint_matches_reference_and_resets_between_requests() {
    let path = std::env::var("PRIVACY_FILTER_CHECKPOINT").expect("set PRIVACY_FILTER_CHECKPOINT");
    let root = Path::new(&path);
    let mut file = std::fs::File::open(root.join("model.safetensors")).unwrap();
    let mut first = [0u8; 8];
    file.read_exact(&mut first).unwrap();
    let length = u64::from_le_bytes(first) as usize;
    let mut header = first.to_vec();
    header.resize(8 + length, 0);
    file.read_exact(&mut header[8..]).unwrap();
    let mut loader = StreamingWeights::new(&header, "130ea7b3a1de096aa5a1c32500d82a8157546c83607d5f71e361df0710fb9d7b").unwrap();
    let plan = loader.entries().to_vec();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    for entry in plan {
        loader.begin(&entry.name).unwrap();
        let mut remaining = (entry.end - entry.start) as usize;
        while remaining > 0 {
            let count = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..count]).unwrap();
            loader.push(&buffer[..count]).unwrap();
            remaining -= count;
        }
        loader.end().unwrap();
    }
    let detector = Detector::new(loader.finish().unwrap(), &std::fs::read_to_string(root.join("config.json")).unwrap(),
                                 &std::fs::read(root.join("tokenizer.json")).unwrap()).unwrap();
    let first = "Hola, soy Lucía Martín y mi correo es lucia@example.com.";
    let result = detector.detect(first).unwrap();
    assert_eq!(result.entities.len(), 2);
    assert_eq!((&*result.entities[0].label, result.entities[0].start, result.entities[0].end), ("PERSON", 10, 22));
    assert_eq!((&*result.entities[1].label, result.entities[1].start, result.entities[1].end), ("EMAIL", 38, 55));
    assert!((result.entities[0].score - 0.447724).abs() < 0.01);
    // This checks parity, not correctness: the reference misclassifies this secret as PERSON.
    let second = detector.detect("Mi contraseña es Almendra123").unwrap();
    assert_eq!(second.entities.len(), 1);
    assert_eq!((&*second.entities[0].label, second.entities[0].start, second.entities[0].end), ("PERSON", 17, 28));
    assert!((second.entities[0].score - 0.218929).abs() < 0.01);
    assert!(detector.detect("💬 Contacta a demo@example.org.").unwrap().entities.is_empty());
    let repeated = detector.detect(first).unwrap();
    assert_eq!(serde_json::to_string(&result).unwrap(), serde_json::to_string(&repeated).unwrap());
}
