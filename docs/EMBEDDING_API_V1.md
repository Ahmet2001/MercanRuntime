# Embedding API v1

The embedding API returns one pooled vector per text from the final normalized
hidden states (`result_norm`, the tensor before the output projection). It
borrows a loaded model and owns one embedding context, so it can be used next
to chat sessions and generation requests on the same weights.

```c
mercan_context_params cp = mercan_context_default_params();
cp.n_ctx = 512;  // tokens per text; longer texts are truncated
mercan_embedder * e = mercan_embedder_create_v1(model, cp, MERCAN_POOLING_MEAN);
int32_t n = mercan_embedder_size_v1(e);
float * v = malloc(sizeof(float) * n);
if (mercan_embed_v1(e, text, strlen(text), v, n) != n) puts(mercan_last_error());
mercan_embedder_free_v1(e);
```

- `MERCAN_POOLING_MEAN` averages all token states; `MERCAN_POOLING_LAST` takes
  the final token's state.
- Vectors are not L2-normalized; normalize before cosine similarity.
- Each call clears the context first, so results do not depend on earlier
  texts. Calls on one handle are serialized; use several handles for parallel
  work.
- The whole text is decoded as one micro-batch (`n_batch = n_ubatch = n_ctx`),
  as pooling requires.
- Only backend-managed architectures (such as `nedolm`) are supported.
  Plugin-managed graphs fail at `mercan_embedder_create_v1` with an error.
- Free every embedder before freeing its model.

## Retrieval quality

Mercan-0.8B is a generative model, not an embedding model. On the Turkish XQuAD
split (240 paragraphs, 1190 questions; find the paragraph a question belongs
to) with `Mercan-0.8B-Final-Q4_K_M`:

| Method | Recall@1 | Recall@5 |
|---|---|---|
| BM25, five-letter prefix terms (EthosoftLib `rag`) | 0.915 | 0.981 |
| Mercan mean pooling | 0.480 | 0.739 |
| Mercan last-token pooling | 0.353 | 0.602 |

XQuAD questions share many words with their paragraphs, which favours BM25.
Use these vectors to find passages that share no term with the query, not to
re-rank lexical matches; EthosoftLib's `rag.Index` does exactly that.
