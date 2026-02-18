use serde_json::Value;

const IDENTITY_REPLACEMENT: &str = "Claude Code";
const GENERIC_UPSTREAM_ERROR: &str = "上游服务暂时不可用，请稍后重试";

const BANNED_CANONICAL_TERMS: &[&str] = &["kiro", "kiroide", "kirors", "kiroapi", "kiroprovider"];
const BANNED_RAW_TERMS: &[&str] = &[
    "kiro",
    "kiro ide",
    "kiro-rs",
    "kiro api",
    "kiro provider",
    "i am kiro",
    "i'm kiro",
    "我是kiro",
    "我是 kiro",
];

fn fold_fullwidth(c: char) -> char {
    match c {
        '\u{FF10}'..='\u{FF19}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
        '\u{FF21}'..='\u{FF3A}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
        '\u{FF41}'..='\u{FF5A}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
        _ => c,
    }
}

fn is_ignored_separator(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '-' | '_' | '.' | ':' | '/' | '\\' | '\'' | '"' | '`' | '+' | '|' | '!' | '?'
        )
        || matches!(c, '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}')
}

fn canonicalize(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for raw in input.chars() {
        let c = fold_fullwidth(raw).to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !is_ignored_separator(c) {
            out.push(c);
        }
    }
    out
}

fn normalize_match_char(c: char) -> Option<char> {
    let n = fold_fullwidth(c).to_ascii_lowercase();
    if n.is_ascii_alphanumeric() {
        Some(n)
    } else {
        None
    }
}

fn match_fuzzy_token(chars: &[char], start: usize, token: &[char]) -> Option<usize> {
    if start >= chars.len() {
        return None;
    }

    let mut j = start;
    let mut matched = 0usize;
    let mut started = false;

    while j < chars.len() && matched < token.len() {
        let c = chars[j];
        if let Some(nc) = normalize_match_char(c) {
            if nc == token[matched] {
                matched += 1;
                started = true;
                j += 1;
                continue;
            }
            break;
        }

        if started && is_ignored_separator(c) {
            j += 1;
            continue;
        }
        break;
    }

    if matched == token.len() {
        Some(j)
    } else {
        None
    }
}

fn replace_fuzzy_token(input: &str, token: &str, replacement: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let token_chars: Vec<char> = token.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;

    while i < chars.len() {
        if let Some(end) = match_fuzzy_token(&chars, i, &token_chars) {
            out.push_str(replacement);
            i = end;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }

    out
}

fn replace_ascii_case_insensitive(input: &str, needle: &str, replacement: &str) -> String {
    if input.is_empty() || needle.is_empty() {
        return input.to_string();
    }

    let hay = input.as_bytes();
    let ned = needle.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;

    while i < hay.len() {
        if i + ned.len() <= hay.len()
            && hay[i..i + ned.len()]
                .iter()
                .zip(ned.iter())
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
        {
            out.push_str(replacement);
            i += ned.len();
        } else {
            let ch = input[i..]
                .chars()
                .next()
                .expect("valid utf-8 boundary while scanning");
            out.push(ch);
            i += ch.len_utf8();
        }
    }

    out
}

pub fn contains_banned_identity(input: &str) -> bool {
    let lowered = input.to_lowercase();
    if BANNED_RAW_TERMS.iter().any(|t| lowered.contains(t)) {
        return true;
    }

    let canonical = canonicalize(input);
    BANNED_CANONICAL_TERMS
        .iter()
        .any(|t| canonical.contains(t))
}

pub fn sanitize_identity_text(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    let mut out = input.to_string();

    let direct_replacements = [
        ("kiro ide", IDENTITY_REPLACEMENT),
        ("kiro-rs", IDENTITY_REPLACEMENT),
        ("kiro api", "Upstream API"),
        ("kiro provider", "upstream provider"),
        ("i am kiro", "I am Claude Code"),
        ("i'm kiro", "I'm Claude Code"),
        ("我是kiro", "我是Claude Code"),
        ("我是 kiro", "我是 Claude Code"),
        ("kiro", IDENTITY_REPLACEMENT),
    ];

    for (needle, replacement) in direct_replacements {
        out = replace_ascii_case_insensitive(&out, needle, replacement);
    }

    out = replace_fuzzy_token(&out, "kiro", IDENTITY_REPLACEMENT);
    out = replace_fuzzy_token(&out, "kiroide", IDENTITY_REPLACEMENT);
    out = replace_fuzzy_token(&out, "kiroapi", "Upstream API");

    if contains_banned_identity(&out) {
        return IDENTITY_REPLACEMENT.to_string();
    }

    out
}

pub fn sanitize_json_value(v: &mut Value) {
    match v {
        Value::String(s) => {
            *s = sanitize_identity_text(s);
        }
        Value::Array(arr) => {
            for item in arr {
                sanitize_json_value(item);
            }
        }
        Value::Object(map) => {
            for (_, value) in map.iter_mut() {
                sanitize_json_value(value);
            }
        }
        _ => {}
    }
}

pub fn sanitize_error_message_for_user(_msg: &str) -> String {
    GENERIC_UPSTREAM_ERROR.to_string()
}

#[derive(Debug, Clone)]
pub struct StreamTextSanitizer {
    carry: String,
    keep_chars: usize,
}

impl StreamTextSanitizer {
    pub fn new(keep_chars: usize) -> Self {
        Self {
            carry: String::new(),
            keep_chars: keep_chars.max(4),
        }
    }

    pub fn push_emit(&mut self, piece: &str) -> String {
        if piece.is_empty() {
            return String::new();
        }

        self.carry.push_str(piece);
        let total_chars = self.carry.chars().count();
        if total_chars <= self.keep_chars {
            return String::new();
        }

        let emit_chars = total_chars - self.keep_chars;
        let emit: String = self.carry.chars().take(emit_chars).collect();
        let tail: String = self.carry.chars().skip(emit_chars).collect();
        self.carry = tail;

        sanitize_identity_text(&emit)
    }

    pub fn flush(&mut self) -> String {
        if self.carry.is_empty() {
            return String::new();
        }

        let out = sanitize_identity_text(&self.carry);
        self.carry.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_banned_identity_with_separators() {
        assert!(contains_banned_identity("K i-r_o"));
    }

    #[test]
    fn test_sanitize_identity_text_basic() {
        let s = sanitize_identity_text("你好，我是 Kiro API 助手");
        assert!(!contains_banned_identity(&s));
    }
}
