use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::slice;
use std::sync::OnceLock;

use nedo_tokenizer::{SurfaceVocabulary, Tokenizer, TokenizerConfig, TokenizerMode};
use sha2::{Digest, Sha256};

const EMBEDDED_VOCAB: &[u8] = include_bytes!("../../NedoTokenizer/assets/surface-vocab.bin");
const EMBEDDED_SHA256: &str = "72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600";

struct State {
    tokenizer: Tokenizer<'static>,
    vocabulary: SurfaceVocabulary,
    sha256: String,
    sha256_c: CString,
}

static STATE: OnceLock<Result<State, String>> = OnceLock::new();

thread_local! {
    static OUTPUT_IDS: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
}

fn make_state(bytes: &[u8], expected_sha: &str) -> Result<State, String> {
    let actual_sha = format!("{:x}", Sha256::digest(bytes));
    if !expected_sha.is_empty() && actual_sha != expected_sha {
        return Err(format!(
            "NDSRF004 vocab SHA mismatch: expected={expected_sha} actual={actual_sha}"
        ));
    }

    let tokenizer = Tokenizer::embedded(TokenizerConfig {
        mode: TokenizerMode::Auto,
        max_sentence_tokens: 512,
        max_fallback_chars: 48,
        contextual_disambiguation: true,
        detect_unmarked_code: true,
    })
    .map_err(|e| format!("NDSRF004 tokenizer init failed: {e}"))?;

    let vocabulary = SurfaceVocabulary::from_bytes(bytes)
        .map_err(|e| format!("NDSRF004 vocab init failed: {e}"))?;
    if vocabulary.len() != 32_000 {
        return Err(format!("NDSRF004 vocab size mismatch: {}", vocabulary.len()));
    }

    let sha256_c = CString::new(actual_sha.as_str())
        .map_err(|_| "invalid SHA string".to_string())?;

    Ok(State {
        tokenizer,
        vocabulary,
        sha256: actual_sha,
        sha256_c,
    })
}

fn state() -> Result<&'static State, &'static str> {
    match STATE.get_or_init(|| make_state(EMBEDDED_VOCAB, EMBEDDED_SHA256)) {
        Ok(s) => Ok(s),
        Err(e) => {
            eprintln!("{e}");
            Err("NDSRF004 state initialization failed")
        }
    }
}

/// Installs the exact surface vocabulary carried by a .mercan file.
///
/// Return codes:
///   0  success / same vocab already active
///  -1  invalid arguments
///  -2  invalid expected SHA string
///  -3  vocab parse or SHA validation failure
///  -4  a different tokenizer vocab is already active in this process
#[no_mangle]
pub unsafe extern "C" fn nedo004_set_vocab(
    data: *const u8,
    len: usize,
    expected_sha: *const std::ffi::c_char,
) -> i32 {
    if data.is_null() || len == 0 || expected_sha.is_null() {
        return -1;
    }

    let expected = match CStr::from_ptr(expected_sha).to_str() {
        Ok(v) if v.len() == 64 => v,
        _ => return -2,
    };

    if let Some(existing) = STATE.get() {
        return match existing {
            Ok(st) if st.sha256 == expected => 0,
            Ok(_) => -4,
            Err(_) => -3,
        };
    }

    let bytes = slice::from_raw_parts(data, len);
    let new_state = match make_state(bytes, expected) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return -3;
        }
    };

    match STATE.set(Ok(new_state)) {
        Ok(()) => 0,
        Err(candidate) => {
            let candidate_sha = match candidate {
                Ok(st) => st.sha256,
                Err(_) => return -3,
            };
            match STATE.get() {
                Some(Ok(st)) if st.sha256 == candidate_sha => 0,
                _ => -4,
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn nedo004_vocab_sha256() -> *const std::ffi::c_char {
    match state() {
        Ok(st) => st.sha256_c.as_ptr(),
        Err(_) => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn nedo004_encode(
    data: *const u8,
    len: usize,
    out_len: *mut usize,
) -> *const u16 {
    if out_len.is_null() || (len != 0 && data.is_null()) {
        return std::ptr::null();
    }
    *out_len = 0;

    let input: &[u8] = if len == 0 { &[] } else { slice::from_raw_parts(data, len) };
    let st = match state() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let document = match st.tokenizer.tokenize(input.to_vec()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("NDSRF004 tokenize failed: {e}");
            return std::ptr::null();
        }
    };
    let encoded = match st.vocabulary.encode_document(&document, false) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("NDSRF004 surface encode failed: {e}");
            return std::ptr::null();
        }
    };
    if encoded.ids.len() < 2 {
        eprintln!("NDSRF004 encoded document lacks BOS/EOS boundaries");
        return std::ptr::null();
    }

    OUTPUT_IDS.with(|cell| {
        let mut ids = cell.borrow_mut();
        ids.clear();
        ids.extend_from_slice(&encoded.ids[1..encoded.ids.len() - 1]);
        *out_len = ids.len();
        if ids.is_empty() {
            std::ptr::NonNull::<u16>::dangling().as_ptr()
        } else {
            ids.as_ptr()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_vocab_contract_loads() {
        let s = state().unwrap();
        assert_eq!(s.vocabulary.len(), 32_000);
        assert_eq!(s.sha256, EMBEDDED_SHA256);
    }
}

// Independent NDSRF004 ABI for SDK clients that do not have a loaded LLM.
// Return codes: 0 success, -1 bad arguments, -2 state unavailable,
// -3 encode/decode failure, -4 output buffer too small.
// out_len is always set to the required size when available.
#[no_mangle]
pub extern "C" fn nedo004_vocab_size() -> u32 {
    state().map(|s| s.vocabulary.len() as u32).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn nedo004_encode_copy(
    data: *const u8,
    len: usize,
    output: *mut u16,
    capacity: usize,
    out_len: *mut usize,
) -> i32 {
    if out_len.is_null() || (len != 0 && data.is_null()) ||
        (capacity != 0 && output.is_null()) {
        return -1;
    }
    *out_len = 0;
    let mut count = 0usize;
    let ids = nedo004_encode(data, len, &mut count);
    if ids.is_null() {
        return -2;
    }
    *out_len = count;
    if capacity < count {
        return -4;
    }
    if count > 0 {
        std::ptr::copy_nonoverlapping(ids, output, count);
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn nedo004_decode_copy(
    ids: *const u16,
    len: usize,
    output: *mut u8,
    capacity: usize,
    out_len: *mut usize,
) -> i32 {
    if out_len.is_null() || (len != 0 && ids.is_null()) ||
        (capacity != 0 && output.is_null()) {
        return -1;
    }
    *out_len = 0;
    let st = match state() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    let tokens: &[u16] = if len == 0 { &[] } else { slice::from_raw_parts(ids, len) };
    let bytes = match st.vocabulary.decode_ids(tokens) {
        Ok(v) => v,
        Err(_) => return -3,
    };
    *out_len = bytes.len();
    if capacity < bytes.len() {
        return -4;
    }
    if !bytes.is_empty() {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, bytes.len());
    }
    0
}

#[cfg(test)]
mod independent_ffi_tests {
    use super::*;

    #[test]
    fn byte_exact_round_trip() {
        let raw = "Merhaba, nasılsın? İstanbul 😀".as_bytes();
        let mut count = 0_usize;
        let requested = unsafe {
            nedo004_encode_copy(raw.as_ptr(), raw.len(), std::ptr::null_mut(), 0, &mut count)
        };
        assert_eq!(requested, -4);
        assert!(count > 0);
        let mut ids = vec![0u16; count];
        assert_eq!(
            unsafe { nedo004_encode_copy(raw.as_ptr(), raw.len(), ids.as_mut_ptr(), ids.len(), &mut count) },
            0,
        );
        let mut out_size = 0_usize;
        assert_eq!(
            unsafe { nedo004_decode_copy(ids.as_ptr(), ids.len(), std::ptr::null_mut(), 0, &mut out_size) },
            -4,
        );
        let mut out = vec![0u8; out_size];
        assert_eq!(
            unsafe { nedo004_decode_copy(ids.as_ptr(), ids.len(), out.as_mut_ptr(), out.len(), &mut out_size) },
            0,
        );
        assert_eq!(out, raw);
        assert_eq!(nedo004_vocab_size(), 32_000);
        assert_eq!(unsafe { CStr::from_ptr(nedo004_vocab_sha256()) }.to_str().unwrap(), EMBEDDED_SHA256);
    }

    #[test]
    fn bad_decode_id_fails_without_crashing() {
        let mut count = 0_usize;
        let invalid = [u16::MAX];
        assert_eq!(
            unsafe { nedo004_decode_copy(invalid.as_ptr(), 1, std::ptr::null_mut(), 0, &mut count) },
            -3,
        );
    }
}
