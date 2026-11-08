#include "mercan_internal.hpp"

#include <mutex>

void mercan_ensure_builtin_plugins(void) {
    static std::once_flag once;
    std::call_once(once, [] {
        mercan_register_nedolm_architecture();
        mercan_register_ndsurf004_tokenizer();
        mercan_register_lfm3_architecture();
        mercan_register_gguf_vocab_tokenizer();
    });
}
