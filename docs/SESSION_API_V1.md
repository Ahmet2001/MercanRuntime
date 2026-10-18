# Persistent session API v1 (MercanRuntime 0.4.0)

The session API adds owned context/KV state while preserving generation API v1
and the frozen ABI 1 parameter layouts. A session borrows its model. Free every
session and generation request before freeing the model/backend.

1. `mercan_session_create_v1(model, context_params)` creates a session. Context
   allocation is lazy, on its first running request.
2. `mercan_session_chat_begin_v1` accepts the complete canonical transcript and
   role headers; `mercan_session_generate_begin_v1` accepts a complete raw prompt.
3. Run the returned handle with `mercan_generation_run_v1` and its existing
   callback/cancellation API.
4. After successfully consuming the output, explicitly call
   `mercan_generation_commit_session_v1` before freeing the generation handle.
5. `mercan_session_reset_v1` clears cached state and releases its context.
   `mercan_session_free_v1` releases the session itself.

Only one request may be outstanding per session, including a completed request
that has not been freed. Reset, stats and free reject busy sessions. Serialize
calls per handle; only generation cancellation may run concurrently with its
worker. The caller owns the transcript: the native session caches token state,
and never silently records or drops application messages.

For each turn the engine compares the newly tokenized prompt against the tokens
actually decoded into the retained context. It reuses only the exact common
prefix and truncates a changed suffix. At least one input token is decoded to
refresh next-token logits. Tokenization boundaries, escaped delimiters, omitted
stop tokens and the final output token are therefore reconciled against the
real next prompt rather than assuming that appending strings preserves tokens.
Sampling state and repetition penalties are initialized separately for each turn.

Generic graph caches rewind their valid position and overwrite stale suffixes.
Llama-managed contexts remove the suffix through the memory API. Backends that
cannot truncate partial state, or have evicted sliding-window history, fall back
to a clean context and full prefill. Context capacity is checked before modifying
an existing cache; excessive prompt plus output budget raises an error. No
automatic history deletion or context shifting is performed.

A failed, cancelled or uncommitted request discards its touched context. Its
next turn performs a full prefill, preventing partially consumed output from
becoming authoritative history. A pre-cancelled request that never touched the
cache leaves the previous cache intact.

`mercan_session_stats_json_v1` returns a borrowed thread-local UTF-8 snapshot
until the next stats call on that thread:

- `cached_tokens`: tokens currently represented by native KV state;
- `reused_tokens`: input-prefix tokens reused by the last completed request;
- `prefill_tokens`: input tokens decoded by that request;
- `context_allocations`: successful context creations over the session lifetime;
- `context_size`: actual allocated capacity, or the configured capacity if idle
  without an allocated context.

Reset clears token/reuse counters but retains the lifetime allocation counter.
KV reuse avoids repeated prefill; it does not increase the context window or
guarantee bitwise equality across different hardware or execution batch shapes.
The interactive CLI uses one such session for its transcript.
