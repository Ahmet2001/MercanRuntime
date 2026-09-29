import Foundation

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
    /// The synthetic tool call and tool result are inference-only. They are not
    /// persisted as visible conversation messages.
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

        // Show the real user message immediately. Search/tool messages stay
        // hidden and are used only for the inference prompt below.
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

        // The search phase is over. ContentView can now replace the network
        // activity indicator with the normal local-model thinking UI.
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
        Web araması uygulama tarafından kullanıcı isteği üzerine zaten yapıldı. Aşağıdaki araç sonucunu güncel kaynak bağlamı olarak kullan. Yeni bir tool call üretme. Araç sonucunda desteklenmeyen ayrıntıları uydurma; kanıt yetersizse bunu açıkça belirt. Kullanıcının sorusunu doğrudan cevapla.
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

        // Persisted conversation contains only user-visible turns.
        for message in messages {
            inferenceMessages.append((
                role: message.isUser ? "user" : "assistant",
                content: message.content
            ))
        }

        // Forced preflight search is represented exactly as a synthetic tool
        // turn followed by an `araç` message. MercanRuntime maps `tool` -> `araç`.
        inferenceMessages.append((role: "assistant", content: syntheticToolCall))
        inferenceMessages.append((role: "tool", content: toolBody))

        // The regular LlamaState engine is private by design. To avoid keeping
        // two model instances alive, temporarily unload it while this explicit
        // tool-grounded turn runs, then restore it afterwards.
        await suspendModelForSpeech()

        let webEngine = MercanRuntimeEngine()
        let started = Date()
        generatedTokenCount = 0
        currentResponse = ""
        isThinking = true
        isGenerating = true

        let filter = SpecialTokenFilter()
        let thinkStripper = ThinkTagStripper()
        var rawParts: [String] = []
        var displayParts: [String] = []

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

            // Keep the current user + synthetic assistant/tool tail intact while
            // dropping only older visible conversation turns if context is tight.
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

            try await webEngine.generateNext(messages: inferenceMessages)

            while await !webEngine.isComplete {
                guard let token = try await webEngine.streamToken() else { continue }
                generatedTokenCount += 1
                rawParts.append(token)

                let filtered = filter.process(token)
                if !filtered.isEmpty {
                    let display = thinkStripper.process(filtered)
                    if !display.isEmpty {
                        displayParts.append(display)
                        currentResponse = displayParts.joined()
                        isThinking = false
                    }
                }
            }

            let filteredTail = filter.flush()
            if !filteredTail.isEmpty {
                let displayTail = thinkStripper.process(filteredTail) + thinkStripper.flush()
                if !displayTail.isEmpty {
                    displayParts.append(displayTail)
                }
            } else {
                let displayTail = thinkStripper.flush()
                if !displayTail.isEmpty {
                    displayParts.append(displayTail)
                }
            }

            let raw = rawParts.joined().trimmingCharacters(in: .whitespacesAndNewlines)
            let display = displayParts.joined().trimmingCharacters(in: .whitespacesAndNewlines)
            let saved = raw.isEmpty ? display : raw

            if !saved.isEmpty {
                messages.append(ChatMessage(content: saved, isUser: false, timestamp: Date()))
            }
            currentResponse = ""

            let entropy = await webEngine.averageEntropy
            modelConfidence = max(0, min(1, 1.0 - (entropy / 12.0)))
            saveCurrentConversation()
        } catch {
            currentResponse = ""
            let errorText = "Web araması tamamlandı ancak yerel model yanıtı üretilemedi: \(error.localizedDescription)"
            messages.append(ChatMessage(content: errorText, isUser: false, timestamp: Date()))
            saveCurrentConversation()
        }

        await webEngine.deinitialize()

        let duration = max(Date().timeIntervalSince(started), 0.001)
        lastGenerationDuration = duration
        lastGenerationTokensPerSecond = Double(generatedTokenCount) / duration
        isThinking = false
        isGenerating = false

        // Restore the normal conversation engine for the next non-web turn.
        await resumeModelAfterSpeech()
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
