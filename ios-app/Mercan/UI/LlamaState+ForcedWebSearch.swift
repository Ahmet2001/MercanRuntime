import Foundation

@MainActor
private final class ForcedWebSearchRuntimeState {
    static let shared = ForcedWebSearchRuntimeState()
    var engine: MercanRuntimeEngine?
    var cancelled = false
}

@MainActor
extension LlamaState {
    /// Explicit Web Search flow selected by the user in the composer.
    ///
    /// The app performs search before any model generation, enriches the first
    /// results with readable page text, then feeds the local model the same
    /// structural sequence it learned during tool-use training:
    ///
    ///   kullanici -> asistan(tool_call JSON) -> araç(web evidence) -> asistan
    ///
    /// Synthetic tool turns are inference-only. If the small local model emits
    /// another tool-call instead of a final answer, that JSON is intercepted and
    /// never shown or persisted. The same evidence is returned as another tool
    /// result and the model gets another final-answer pass. A direct grounded
    /// fallback is used if it keeps requesting the already-completed search.
    func completeWithForcedWebSearch(
        text rawText: String,
        onSearchCompleted: @escaping @MainActor () -> Void = {}
    ) async {
        guard !isGenerating else { return }

        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        if !isModelLoaded {
            guard await ensureModelLoaded() else {
                onSearchCompleted()
                return
            }
        }

        guard let modelURL = forcedSearchModelURL() else {
            modelLoadError = String(localized: "No model file found on device.")
            onSearchCompleted()
            return
        }

        // Only the real user message is visible/persistent.
        let userMessage = ChatMessage(content: text, isUser: true, timestamp: Date())
        messages.append(userMessage)
        saveCurrentConversation()

        let webService = WebSearchService()
        let evidence: [GroundedWebSearchEvidence]
        do {
            evidence = try await webService.searchWithPageContent(
                query: text,
                maxResults: 5,
                maxPages: 3
            )
        } catch {
            evidence = []
        }

        onSearchCompleted()

        let callID = "preflight_web_search"
        let syntheticToolCall = WebSearchService.forcedToolCallJSON(
            query: text,
            callID: callID
        )
        let toolBody = WebSearchService.groundedToolResult(
            query: text,
            callID: callID,
            evidence: evidence
        )

        var inferenceMessages: [(role: String, content: String)] = []
        let groundingInstruction = """
        Web araması uygulama tarafından kullanıcı isteği üzerine zaten tamamlandı. Son araç mesajındaki web kanıtlarını kullanarak şimdi kullanıcının sorusuna doğrudan nihai cevap ver. Yeni bir tool call, JSON veya araç isteği üretme. Araç sonucunda desteklenmeyen ayrıntıları uydurma; kanıt yetersizse bunu açıkça belirt.
        """
        let trimmedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        inferenceMessages.append((
            role: "system",
            content: trimmedSystem.isEmpty
                ? groundingInstruction
                : trimmedSystem + "\n\n" + groundingInstruction
        ))

        if let transcript = await loadResolvedTranscriptText(), !transcript.isEmpty {
            inferenceMessages.append((
                role: "system",
                content: """
                Kullanıcının eklediği video transkripti aşağıdadır. İlgili olduğunda bu yerel bağlamı da kullan.

                TRANSKRİPT:
                \(String(transcript.prefix(12_000)))
                """
            ))
        }

        if let document = attachedDocument {
            inferenceMessages.append((
                role: "system",
                content: """
                Kullanıcının eklediği belge: \(document.name)
                İlgili olduğunda aşağıdaki belge bağlamını da kullan.

                BELGE:
                \(String(document.text.prefix(12_000)))
                """
            ))
        }

        // Persisted history contains only user-visible turns.
        for message in messages {
            inferenceMessages.append((
                role: message.isUser ? "user" : "assistant",
                content: message.content
            ))
        }

        // The search decision is made by the app, but the model sees the
        // canonical training structure.
        inferenceMessages.append((role: "assistant", content: syntheticToolCall))
        inferenceMessages.append((role: "tool", content: toolBody))

        // Temporarily unload the regular engine so only one model instance is
        // resident while the explicit web-grounded turn runs.
        await suspendModelForSpeech()

        let webEngine = MercanRuntimeEngine()
        ForcedWebSearchRuntimeState.shared.engine = webEngine
        ForcedWebSearchRuntimeState.shared.cancelled = false

        let started = Date()
        generatedTokenCount = 0
        currentResponse = ""
        isThinking = true
        isGenerating = true

        do {
            try await webEngine.initialize(
                modelPath: modelURL.path,
                contextSize: contextSize
            )
            await webEngine.setSampling(SamplingConfiguration(
                temperature: Float(temperature),
                topK: Int32(topK),
                topP: Float(topP),
                minP: Float(minP),
                repeatPenalty: Float(repeatPenalty),
                repeatLastN: 64
            ))

            // Preserve the current user + synthetic assistant/tool tail while
            // dropping only older visible turns if context is tight.
            let budget = Int(Double(contextSize) * 0.88)
            var tokenCount = await webEngine.countTokens(for: inferenceMessages)
            while tokenCount > budget && inferenceMessages.count > 4 {
                let protectedTailStart = max(0, inferenceMessages.count - 3)
                var removeIndex = 0
                while removeIndex < protectedTailStart,
                      inferenceMessages[removeIndex].role == "system" {
                    removeIndex += 1
                }
                guard removeIndex < protectedTailStart else { break }
                inferenceMessages.remove(at: removeIndex)
                tokenCount = await webEngine.countTokens(for: inferenceMessages)
                contextTruncated = true
            }
            lastPromptTokenCount = tokenCount
            contextTokenCount = tokenCount

            // Buffer each candidate before displaying it. This prevents a
            // repeated synthetic tool-call JSON from ever reaching the UI.
            var finalRaw = ""
            var finalDisplay = ""
            var toolRetryCount = 0
            let maximumToolRetries = 2

            while !ForcedWebSearchRuntimeState.shared.cancelled && !Task.isCancelled {
                await webEngine.clearGenerationState()
                try await webEngine.generateNext(messages: inferenceMessages)

                var candidateParts: [String] = []
                while await !webEngine.isComplete,
                      !ForcedWebSearchRuntimeState.shared.cancelled,
                      !Task.isCancelled {
                    guard let token = try await webEngine.streamToken() else { continue }
                    generatedTokenCount += 1
                    candidateParts.append(token)
                }

                if ForcedWebSearchRuntimeState.shared.cancelled || Task.isCancelled {
                    await webEngine.stop()
                    break
                }

                let candidateRaw = candidateParts.joined()
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                if let repeatedCalls = WebSearchService.parseToolCalls(from: candidateRaw),
                   !repeatedCalls.isEmpty {
                    // Never display or save the repeated tool JSON. The search
                    // has already happened, so return the same fresh evidence.
                    if toolRetryCount < maximumToolRetries {
                        inferenceMessages.append((role: "assistant", content: candidateRaw))
                        inferenceMessages.append((role: "tool", content: toolBody))
                        toolRetryCount += 1
                        continue
                    }

                    // Last-resort deterministic grounding path. This mirrors the
                    // successful direct-RAG prompt used in runtime validation,
                    // while still keeping the user's visible message unchanged.
                    inferenceMessages = [
                        (
                            role: "system",
                            content: "Web araması tamamlandı. Yalnızca nihai cevabı yaz. JSON, tool call veya araç çağrısı yazma. Web kanıtlarında olmayan ayrıntıları uydurma."
                        ),
                        (
                            role: "user",
                            content: """
                            SORU:
                            \(text)

                            WEB ARAMA KANITLARI:
                            \(toolBody)

                            Yukarıdaki kanıtlara dayanarak soruyu doğrudan cevapla.
                            """
                        )
                    ]
                    toolRetryCount += 1
                    continue
                }

                let filter = SpecialTokenFilter()
                let stripper = ThinkTagStripper()
                let filtered = filter.process(candidateRaw) + filter.flush()
                let display = (stripper.process(filtered) + stripper.flush())
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                // Empty/non-answer generations get one direct grounded retry.
                if display.isEmpty && toolRetryCount <= maximumToolRetries {
                    inferenceMessages = [
                        (
                            role: "system",
                            content: "Web araması tamamlandı. Yalnızca nihai cevabı yaz; JSON veya tool call üretme."
                        ),
                        (
                            role: "user",
                            content: "SORU:\n\(text)\n\nWEB ARAMA KANITLARI:\n\(toolBody)"
                        )
                    ]
                    toolRetryCount += 1
                    continue
                }

                finalRaw = candidateRaw
                finalDisplay = display
                break
            }

            if !ForcedWebSearchRuntimeState.shared.cancelled && !Task.isCancelled {
                let saved = finalDisplay.isEmpty ? finalRaw : finalDisplay
                if !saved.isEmpty,
                   WebSearchService.parseToolCalls(from: saved) == nil {
                    // Only a human-readable final answer can enter chat history.
                    currentResponse = saved
                    messages.append(ChatMessage(content: saved, isUser: false, timestamp: Date()))
                    saveCurrentConversation()
                }

                let entropy = await webEngine.averageEntropy
                modelConfidence = max(0, min(1, 1.0 - (entropy / 12.0)))
            }
        } catch {
            if !ForcedWebSearchRuntimeState.shared.cancelled && !Task.isCancelled {
                currentResponse = ""
                let errorText = "Web araması tamamlandı ancak yerel model yanıtı üretilemedi: \(error.localizedDescription)"
                messages.append(ChatMessage(content: errorText, isUser: false, timestamp: Date()))
                saveCurrentConversation()
            }
        }

        await webEngine.deinitialize()
        ForcedWebSearchRuntimeState.shared.engine = nil
        ForcedWebSearchRuntimeState.shared.cancelled = false

        let duration = max(Date().timeIntervalSince(started), 0.001)
        lastGenerationDuration = duration
        lastGenerationTokensPerSecond = Double(generatedTokenCount) / duration
        currentResponse = ""
        isThinking = false
        isGenerating = false

        // Restore the normal conversation engine for the next non-web turn.
        await resumeModelAfterSpeech()
    }

    func cancelForcedWebSearchGeneration() async {
        guard let engine = ForcedWebSearchRuntimeState.shared.engine else { return }
        ForcedWebSearchRuntimeState.shared.cancelled = true
        await engine.stop()
        currentResponse = ""
        isThinking = false
    }

    private func forcedSearchModelURL() -> URL? {
        let documents = getDocumentsDirectory()

        if !currentModelName.isEmpty,
           let current = downloadedModels.first(where: {
               URL(fileURLWithPath: $0.filename)
                   .deletingPathExtension()
                   .lastPathComponent == currentModelName
           }) {
            let url = documents.appendingPathComponent(current.filename)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        for model in downloadedModels {
            let url = documents.appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        return nil
    }
}
