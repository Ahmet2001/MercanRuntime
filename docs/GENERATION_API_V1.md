# Native generation and chat API v1

MercanRuntime 0.2.0 adds a shared generation engine to `libmercan`. The CLI and
EthosoftLib 0.6.0 both use it. Model/context ABI 1 remains unchanged.

The engine owns tokenization, batched prefill, sampling, stopping and the token
decode loop. Applications supply a borrowed model and receive output bytes via
a callback. Each request owns a fresh context; cross-turn KV reuse is not implied.

```cpp
#include "mercan.h"
#include <cstdio>

int receive(const char * bytes, size_t size, void *) {
    std::fwrite(bytes, 1, size, stdout);
    return 1; // return 0 to cancel
}

// After backend initialization and model loading:
void reply(mercan_model * model) {
    mercan_generation_params_v1 params{};
    mercan_generation_default_params_v1(&params, sizeof(params));
    params.max_tokens = 128;
    params.temperature = 0;
    const char question[] = "Merhaba";
    mercan_chat_message_v1 messages[] = {{"user", question, sizeof(question) - 1}};
    auto * request = mercan_chat_begin_v1(model, messages, 1,
        nullptr, nullptr, nullptr, mercan_context_default_params(), &params);
    if (!request) { std::fprintf(stderr, "%s\n", mercan_last_error()); return; }
    int status = mercan_generation_run_v1(request, receive, nullptr);
    if (status < 0) std::fprintf(stderr, "%s\n", mercan_last_error());
    mercan_generation_free_v1(request);
}
```

Use `mercan_generate_begin_v1` for an already formatted, length-delimited raw
prompt. `mercan_chat_begin_v1` accepts canonical system/user/assistant messages;
null header arguments select sistem/kullanici/asistan. No default system prompt
or conversation history is inserted. Literal `<|` delimiters inside content are
escaped with a zero-width space. Raw generation is available for applications
that need to supply structural tokens themselves.

## Lifetime, threading and errors

- `begin` copies/tokenizes the input and allocates a context. The caller may
  release message buffers when it returns, but must retain the model.
- Call `run` exactly once per request. It performs the complete native loop.
  Do not run/free the same request concurrently or free its model early.
- `cancel` may be called concurrently from another thread. Cancellation is
  checked between prefill batches and token operations. It does not interrupt
  an in-flight CPU/GPU operation. The callback can also return zero to cancel.
- Wait for `run` to return before `free`. Free requests before their models,
  and all models/contexts before the process-wide backend.
- `run` returns 0 for completion, 1 for cancellation and -1 for an error.
  `finish_reason` distinguishes stop token, token limit, cancellation and error.
  Retrieve/copy `mercan_last_error` on the same thread as the failing call.
- Callbacks borrow their bytes for the call duration only. Copy bytes if needed
  later. UTF-8 characters can span pieces: text consumers need an incremental
  decoder. Callbacks must not throw, free the request/model, or recursively run
  the request. A null callback discards output.

## Sampling contract

Options include maximum tokens, temperature, top-k, top-p, repeat penalty/window
and an optional unsigned 32-bit seed. Zero temperature selects greedy decoding.
Repetition is penalized once per distinct token in the recent **generated**
output window; prompt tokens are excluded. EOS, pad and model-resolved ChatML
boundary tokens stop generation and are exempt from repetition penalties.
Invalid/non-finite options and requests exceeding context capacity fail clearly.

The CLI accepts `--seed`. Identical messages, role headers and sampling options
use the same engine in CLI and SDK. Seeded reproducibility is tested within the
same runtime/backend build; it is not a cross-device floating-point guarantee.

Check `mercan_generation_api_version() == 1` and
`mercan_generation_params_size_v1()` before binding the new parameter layout.
The v1 struct has a `struct_size` prefix. Existing ABI 1 structs are not extended.
Old libraries still support their low-level API, but new high-level SDK calls
require this API and report a rebuild/update error rather than running a second
Python generation loop.

## Verification

Build with `MERCAN_BUILD_SDK_TESTS=ON` and run CTest. Generation tests cover native
sampling, parameter layout, a tiny plugin model, token limits, cancellation and
one-shot request lifecycle. EthosoftLib integration tests additionally exercise
real Mercan chat/raw parity, seeded CLI/SDK parity, UTF-8 streaming, async task
cancellation and context cleanup. Set `MERCAN_TEST_CLI` to include CLI parity.
