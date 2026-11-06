#[test]
#[ignore = "requires the released tokenizer via PRIVACY_FILTER_TOKENIZER"]
fn official_tokenizer_offsets_can_be_mapped_to_original_unicode() {
    let path = std::env::var("PRIVACY_FILTER_TOKENIZER").expect("set PRIVACY_FILTER_TOKENIZER");
    let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
    let input = "💬Ana";
    let encoding = tokenizer.encode(input, false).unwrap();
    println!("offsets={:?}", encoding.get_offsets());
    assert!(!encoding.get_offsets().is_empty());
}
