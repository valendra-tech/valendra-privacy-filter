use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use unicode_categories::UnicodeCategories;

#[derive(Clone, Debug, Serialize)]
pub struct Span {
    #[serde(rename = "type")]
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub value: String,
    pub score: f64,
}

fn parts(label: &str) -> Result<(char, &str), String> {
    if label == "O" { return Ok(('O', "")); }
    let (prefix, category) = label.split_once('-').ok_or("Invalid BIOES label")?;
    let prefix = prefix.chars().next().ok_or("Invalid BIOES label")?;
    if !matches!(prefix, 'B' | 'I' | 'E' | 'S') || category.is_empty() {
        return Err("Invalid BIOES label".into());
    }
    Ok((prefix, category))
}

pub fn viterbi_decode(logits: &[Vec<f32>], names: &[String]) -> Result<Vec<usize>, String> {
    let labels: Vec<_> = names.iter().map(|n| parts(n)).collect::<Result<_, _>>()?;
    if labels.is_empty() || logits.iter().any(|row| row.len() != names.len() || row.iter().any(|x| !x.is_finite())) {
        return Err("Invalid logits".into());
    }
    if logits.is_empty() { return Ok(Vec::new()); }
    let mut state: Vec<f32> = logits[0].iter().zip(&labels).map(|(&x, (p, _))|
        if matches!(p, 'O' | 'B' | 'S') { x } else { f32::NEG_INFINITY }).collect();
    let mut back = Vec::new();
    for emission in &logits[1..] {
        let mut next = vec![f32::NEG_INFINITY; names.len()];
        let mut previous = vec![0; names.len()];
        for (j, &(current, new_type)) in labels.iter().enumerate() {
            for (i, &(prior, old_type)) in labels.iter().enumerate() {
                let allowed = (matches!(prior, 'O' | 'E' | 'S') && matches!(current, 'O' | 'B' | 'S'))
                    || (matches!(prior, 'B' | 'I') && matches!(current, 'I' | 'E') && old_type == new_type);
                if allowed && state[i] > next[j] { next[j] = state[i]; previous[j] = i; }
            }
            next[j] += emission[j];
        }
        state = next;
        back.push(previous);
    }
    let mut best = None;
    for (i, (prefix, _)) in labels.iter().enumerate() {
        if matches!(prefix, 'O' | 'E' | 'S') && state[i].is_finite()
            && best.map_or(true, |j| state[i] > state[j]) { best = Some(i); }
    }
    let mut current = best.ok_or("No valid BIOES path")?;
    let mut path = vec![current];
    for previous in back.iter().rev() { current = previous[current]; path.push(current); }
    path.reverse();
    Ok(path)
}

pub fn spans_from_tags(text: &str, offsets: &[(usize, usize)], path: &[usize], names: &[String],
                       probabilities: &[f64]) -> Result<Vec<Span>, String> {
    if offsets.len() != path.len() || path.len() != probabilities.len() { return Err("Invalid token lengths".into()); }
    let mut spans = Vec::new();
    let mut active: Option<(String, Vec<usize>)> = None;
    for (i, &id) in path.iter().enumerate() {
        let (prefix, category) = parts(names.get(id).ok_or("Unknown label")?)?;
        let complete = match prefix {
            'S' => Some((category.to_string(), vec![i])),
            'B' => { active = Some((category.to_string(), vec![i])); None },
            'I' | 'E' => {
                if let Some((ref name, ref mut ids)) = active {
                    if name == category {
                        ids.push(i);
                        if prefix == 'E' { active.take() } else { None }
                    } else { active = None; None }
                } else { None }
            },
            _ => { active = None; None },
        };
        if let Some((name, ids)) = complete {
            let mut start = ids.iter().map(|&j| offsets[j].0).min().ok_or("Empty span")?;
            let mut end = ids.iter().map(|&j| offsets[j].1).max().ok_or("Empty span")?;
            if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                return Err("Invalid Unicode offsets".into());
            }
            while start < end && text[start..].chars().next().unwrap().is_whitespace() {
                start += text[start..].chars().next().unwrap().len_utf8();
            }
            while end > start && text[..end].chars().next_back().unwrap().is_whitespace() {
                end -= text[..end].chars().next_back().unwrap().len_utf8();
            }
            if start < end {
                let score = (ids.iter().map(|&j| probabilities[j].max(1e-30).ln()).sum::<f64>() / ids.len() as f64).exp();
                spans.push(Span { label: name, start: text[..start].chars().count(), end: text[..end].chars().count(),
                                  value: text[start..end].to_string(), score });
            }
        }
    }
    spans.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.start.cmp(&b.start)).then((b.end-b.start).cmp(&(a.end-a.start))));
    let mut selected: Vec<Span> = Vec::new();
    for span in spans {
        if !selected.iter().any(|other| span.start < other.end && other.start < span.end) { selected.push(span); }
    }
    selected.sort_by_key(|span| (span.start, span.end));
    Ok(selected)
}

fn word_character(c: char) -> bool { c.is_alphanumeric() || c == '_' || c.is_mark() }

fn char_offsets(text: &str, offsets: &[(usize, usize)]) -> Option<Vec<(usize, usize)>> {
    let mut positions = vec![usize::MAX; text.len() + 1];
    for (index, (byte, _)) in text.char_indices().enumerate() { positions[byte] = index; }
    positions[text.len()] = text.chars().count();
    offsets.iter().map(|&(a, b)| {
        if a > b || b > text.len() || positions[a] == usize::MAX || positions[b] == usize::MAX { None }
        else { Some((positions[a], positions[b])) }
    }).collect()
}

fn span_paths(chars: &[char], offsets: &[(usize, usize)], logits: &[Vec<f32>], names: &[String],
              start: usize, end: usize, categories: &BTreeSet<String>) -> Option<(HashMap<String, f64>, f64, usize)> {
    let indices: Vec<usize> = offsets.iter().enumerate().filter_map(|(i, &(a,b))| (a < end && b > start).then_some(i)).collect();
    if indices.is_empty() || indices.iter().enumerate().any(|(n, &i)| i != indices[0] + n) { return None; }
    let first = offsets[*indices.first()?];
    let last = offsets[*indices.last()?];
    if first.0 > start || last.1 < end || chars[first.0..start].iter().any(|c| !c.is_whitespace())
        || chars[end..last.1].iter().any(|c| !c.is_whitespace()) { return None; }
    let labels: HashMap<&str, usize> = names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    let outside_id = *labels.get("O")?;
    let mut scores: HashMap<String, f64> = categories.iter().map(|c| (c.clone(), 0.0)).collect();
    let mut outside = 0.0;
    for (position, &token) in indices.iter().enumerate() {
        let prefix = if indices.len() == 1 { 'S' } else if position == 0 { 'B' } else if position + 1 == indices.len() { 'E' } else { 'I' };
        let row = logits.get(token)?;
        if row.len() != names.len() || row.iter().any(|v| !v.is_finite()) { return None; }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let normalizer = row.iter().map(|&v| ((v as f64) - max).exp()).sum::<f64>().ln() + max;
        outside += row[outside_id] as f64 - normalizer;
        for category in categories {
            let label = format!("{prefix}-{category}");
            let id = *labels.get(label.as_str())?;
            *scores.get_mut(category)? += row[id] as f64 - normalizer;
        }
    }
    Some((scores, outside, indices.len()))
}

pub fn reconcile_secrets(text: &str, byte_offsets: &[(usize, usize)], logits: &[Vec<f32>], names: &[String],
                         entities: Vec<Span>) -> Result<Vec<Span>, String> {
    if entities.is_empty() || !entities.iter().any(|e| e.label == "SECRET") { return Ok(entities); }
    if logits.len() != byte_offsets.len() { return Ok(entities); }
    let Some(offsets) = char_offsets(text, byte_offsets) else { return Ok(entities) };
    let chars: Vec<char> = text.chars().collect();
    if entities.iter().any(|e| e.start >= e.end || e.end > chars.len()) { return Err("Invalid entity offsets".into()); }
    let mut grouped = Vec::new();
    let mut position = 0;
    while position < entities.len() {
        let mut end_position = position + 1;
        while end_position < entities.len() {
            let boundary = entities[end_position - 1].end;
            if boundary != entities[end_position].start || boundary == 0 || boundary >= chars.len()
                || !word_character(chars[boundary-1]) || !word_character(chars[boundary]) { break; }
            end_position += 1;
        }
        let group = &entities[position..end_position];
        let start = group[0].start;
        let end = group.last().unwrap().end;
        let categories: BTreeSet<String> = group.iter().map(|e| e.label.clone()).collect();
        let whole = chars[start..end].iter().all(|&c| word_character(c))
            && (start == 0 || !word_character(chars[start-1]))
            && (end == chars.len() || !word_character(chars[end]));
        let candidate = if group.len() > 1 && categories.contains("SECRET") && whole {
            span_paths(&chars, &offsets, logits, names, start, end, &categories)
        } else { None };
        if let Some((scores, _, count)) = candidate {
            let secret = scores["SECRET"];
            if secret.is_finite() && scores.iter().all(|(name, &value)| name == "SECRET" || secret > value) {
                grouped.push(Span { label: "SECRET".into(), start, end, value: chars[start..end].iter().collect(), score: (secret / count as f64).exp() });
                position = end_position;
                continue;
            }
        }
        grouped.extend_from_slice(group);
        position = end_position;
    }

    let mut runs: Vec<(usize, usize, f64)> = Vec::new();
    for entity in grouped.iter().filter(|e| e.label == "SECRET") {
        let (mut start, mut end) = (entity.start, entity.end);
        while start > 0 && !chars[start-1].is_whitespace() { start -= 1; }
        while end < chars.len() && !chars[end].is_whitespace() { end += 1; }
        while start < end && "\"'«“‘([{<".contains(chars[start]) { start += 1; }
        while end > start && "\"'»”’)]}>.,;:".contains(chars[end-1]) { end -= 1; }
        if runs.iter().any(|&(a,b,_)| (a,b) == (start,end)) || (start,end) == (entity.start,entity.end)
            || start > entity.start || end < entity.end { continue; }
        let overlapping: Vec<_> = grouped.iter().filter(|e| e.start < end && e.end > start).collect();
        if overlapping.iter().any(|e| e.start < start || e.end > end) { continue; }
        let categories: BTreeSet<String> = overlapping.iter().map(|e| e.label.clone()).collect();
        if let Some((scores, outside, count)) = span_paths(&chars, &offsets, logits, names, start, end, &categories) {
            let secret = scores["SECRET"];
            if secret.is_finite() && secret > outside && scores.iter().all(|(name, &value)| name == "SECRET" || secret > value) {
                runs.push((start, end, (secret / count as f64).exp()));
            }
        }
    }
    if runs.is_empty() { return Ok(grouped); }
    grouped.retain(|item| !runs.iter().any(|&(a,b,_)| item.start < b && item.end > a));
    grouped.extend(runs.into_iter().map(|(start,end,score)| Span { label: "SECRET".into(), start, end, value: chars[start..end].iter().collect(), score }));
    grouped.sort_by_key(|span| (span.start, span.end));
    Ok(grouped)
}
