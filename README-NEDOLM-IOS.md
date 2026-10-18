# NedoLM iOS build payload

Bu paket `Ahmet2001/MercanRuntime` reposuna eklenecek GitHub Actions dosyalarını içerir.

## Gereken iki runtime dosyası

TRUBA'dan aşağıdaki iki dosya `runtime/` altına konmalıdır:

1. `runtime/nedolm-llama.patch`
   - Base: llama.cpp commit `e71b80510c848c00175924ecf3c40333ccae8eb5`
   - NedoLM custom architecture + MorphFFN + SWA + NeoX RoPE + NDSRF004 bridge değişiklikleri.

2. `runtime/libnedo004_ffi.a`
   - Exact NDSRF004 Rust FFI staticlib.
   - **aarch64-apple-ios** için derlenmiş olmalıdır.
   - Tokenizer contract SHA256:
     `72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600`

Model GGUF / checkpoint GitHub reposuna eklenmez.

## Build

Workflow:
`.github/workflows/nedolm-ios.yml`

Mac runner:
`macos-15`

Çıktı:
`NedoLM-Silo-unsigned.ipa`

IPA unsigned'dır. Windows'ta Sideloadly ile Apple ID kullanılarak yeniden imzalanabilir.

## Model

Final runtime hedefi step 6478 modelidir. Model uygulamaya gömülmez; Silo'nun GGUF import/Hugging Face URL akışı kullanılır.

Beklenen final GGUF SHA256:
`780eb3a95468b599cbaf896c894b3a9c28802d35f5271640c1906e7e9f769eed`
