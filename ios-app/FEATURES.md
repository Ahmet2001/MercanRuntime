# Mercan iOS Experience

The iOS app is a local-first Mercan client built on `libmercan`.

## Included

- Mercan branding and app icon
- First-run onboarding
- Local/offline status in the chat header
- Conversation search and pinning
- Copy and regenerate actions for assistant replies
- Model catalog, download, local import and URL import
- Device-aware context recommendation
- Generation presets: Precise, Balanced and Creative
- Advanced temperature, Top-P, Top-K, Min-P and repeat-penalty controls
- LFM3 reasoning On/Off (composer shortcut and Settings; saved on device)
- Expandable reasoning traces in saved chat replies and live streamed reasoning
- Context usage meter
- Last-generation tokens/second metrics
- Local PDF/TXT/Markdown/JSON/CSV document grounding
- System-prompt presets plus custom system prompt
- Runtime diagnostics
- Local inference benchmark
- Privacy explanation
- User-friendly model error card

## LFM reasoning

Load a compatible LFM3 `.mercan` model, then tap the brain button next to the message field or enable **Reasoning (LFM)** in Settings > Generation. The setting is off by default and is preserved between launches. When enabled, Mercan starts the LFM assistant turn with `<think>`; when disabled, it uses the upstream closed-block `<think></think>` prompt. The model may not consistently obey the no-thinking request.

If the model generates a reasoning block, it appears in a collapsible **Reasoning** section above the final answer. Reasoning can be expanded after the response and is expanded while a normal response is streaming. During web search, candidate responses are buffered to prevent tool-call JSON or raw evidence from appearing before validation. Previous turns' reasoning is not replayed into the next inference prompt.

## Document grounding

The document button in the composer accepts text-based documents and PDFs. Text is extracted locally and added to the current chat as grounding context. The extracted document context is cleared when a new or different conversation is opened.

The current safety limit is 120,000 extracted characters, with the inference prompt further constrained by the selected model context window.

## Model format

The iOS catalog is Mercan-format-native. `libmercan` validates the format, runtime ABI, architecture and tokenizer when a model is loaded.

See:

- [PyTorch to Mercan conversion](../docs/PT_TO_MERCAN.md)
- [Mercan Format v1](../spec/MERCAN_FORMAT_V1.md)

## Audio and video

This experience refresh intentionally does not expand the existing audio/video feature set. Existing speech/video implementation files remain separate from the new chat, model-management and document workflows.
