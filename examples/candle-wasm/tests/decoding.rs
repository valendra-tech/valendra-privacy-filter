use valendra_candle_wasm::decode::{reconcile_secrets, spans_from_tags, viterbi_decode, Span};

#[test]
fn viterbi_enforces_bioes_start_end_and_category() {
    let names = ["O", "B-PERSON", "I-PERSON", "E-PERSON", "S-PERSON", "B-EMAIL", "I-EMAIL", "E-EMAIL", "S-EMAIL"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let mut logits = vec![vec![-10.0; names.len()]; 3];
    logits[0][2] = 20.0;
    logits[0][1] = 19.0;
    logits[1][6] = 20.0;
    logits[1][2] = 18.0;
    logits[2][3] = 20.0;
    assert_eq!(viterbi_decode(&logits, &names).unwrap(), vec![1, 2, 3]);
}

#[test]
fn reconstructs_unicode_offsets_and_token_score() {
    let names = ["O", "B-PERSON", "I-PERSON", "E-PERSON", "S-PERSON"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let result = spans_from_tags("💬Ana!", &[(0, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
                                 &[0, 1, 2, 3, 0], &names, &[1.0, 0.9, 0.8, 0.7, 1.0]).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!((result[0].start, result[0].end), (1, 4));
    assert_eq!(result[0].label, "PERSON");
    assert_eq!(result[0].value, "Ana");
    assert!((result[0].score - (0.9_f64 * 0.8 * 0.7).powf(1.0 / 3.0)).abs() < 1e-8);
}

#[test]
fn reconciles_adjacent_person_and_secret_using_the_joint_secret_path() {
    let names = ["O", "B-PERSON", "I-PERSON", "E-PERSON", "S-PERSON", "B-SECRET", "I-SECRET", "E-SECRET", "S-SECRET"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let mut logits = vec![vec![-12.0; names.len()]; 4];
    logits[0][1] = 8.0; logits[0][5] = 7.0;
    logits[1][3] = 8.0; logits[1][6] = 6.0;
    logits[2][5] = 8.0; logits[2][6] = 7.0;
    logits[3][7] = 8.0;
    let spans = vec![
        Span { label: "PERSON".into(), start: 0, end: 2, value: "Ab".into(), score: 0.9 },
        Span { label: "SECRET".into(), start: 2, end: 4, value: "12".into(), score: 0.1 },
    ];
    let merged = reconcile_secrets("Ab12", &[(0,1),(1,2),(2,3),(3,4)], &logits, &names, spans).unwrap();
    assert_eq!(merged.len(), 1);
    assert_eq!((&*merged[0].label, &*merged[0].value), ("SECRET", "Ab12"));
    assert_eq!((merged[0].start, merged[0].end), (0, 4));
    assert!(merged[0].score > 0.0 && merged[0].score < 1.0);
}

#[test]
fn does_not_merge_secrets_across_whitespace() {
    let names = ["O", "B-PERSON", "I-PERSON", "E-PERSON", "S-PERSON", "B-SECRET", "I-SECRET", "E-SECRET", "S-SECRET"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let spans = vec![
        Span { label: "PERSON".into(), start: 0, end: 2, value: "Ab".into(), score: 0.9 },
        Span { label: "SECRET".into(), start: 3, end: 5, value: "12".into(), score: 0.1 },
    ];
    let result = reconcile_secrets("Ab 12", &[(0,1),(1,2),(2,3),(3,4),(4,5)], &vec![vec![0.0; 9]; 5], &names, spans).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn expands_secret_through_punctuation_when_the_joint_path_wins() {
    let names = ["O", "B-SECRET", "I-SECRET", "E-SECRET", "S-SECRET"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let text = "\"key#12\",";
    let offsets: Vec<_> = (0..text.len()).map(|i| (i, i + 1)).collect();
    let mut logits = vec![vec![-12.0; names.len()]; text.len()];
    for row in &mut logits[1..7] {
        row[0] = 6.0; row[1] = 5.0; row[2] = 5.0; row[3] = 5.0;
    }
    for row in &mut logits[5..7] {
        row[0] = -12.0; row[1] = 8.0; row[2] = 8.0; row[3] = 8.0;
    }
    let spans = vec![Span { label: "SECRET".into(), start: 5, end: 7, value: "12".into(), score: 0.5 }];
    let result = reconcile_secrets(text, &offsets, &logits, &names, spans).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!((&*result[0].label, &*result[0].value), ("SECRET", "key#12"));
    assert_eq!((result[0].start, result[0].end), (1, 7));
}
