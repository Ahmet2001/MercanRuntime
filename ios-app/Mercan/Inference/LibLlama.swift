import Foundation
import MercanRuntime

enum LlamaError: Error, LocalizedError {
    case couldNotInitializeContext(path: String, detail: String)
    case decodeFailed(String)
    case tokenizationFailed(String)
    case samplingFailed(String)

    var errorDescription: String? {
        switch self {
        case .couldNotInitializeContext(let path, let detail):
            let name = (path as NSString).lastPathComponent
            return "Could not load \(name) with libmercan. \(detail)"
        case .decodeFailed(let detail):
            return "Mercan inference failed while decoding. \(detail)"
        case .tokenizationFailed(let detail):
            return "Mercan tokenizer failed. \(detail)"
        case .samplingFailed(let detail):
            return "Mercan sampler failed. \(detail)"
        }
    }
}

/// Swift actor adapter over libmercan's stable C runtime API.
///
/// libmercan owns model loading, Mercan format validation, architecture/tokenizer
/// dispatch, KV state, sampling, tokenization, decoding and logits. Swift owns
/// chat formatting and UI-facing streaming state.
actor MercanRuntimeContext {
    private let model: OpaquePointer
    private let context: OpaquePointer
    private let architectureName: String
    private let tokenizerName: String

    var is_done = false
    private var nCur: Int32 = 0
    private let nLen: Int32

    /// Mirrors the exact decoded token sequence currently retained by libmercan.
    /// It includes generated tokens that were actually decoded into the KV cache.
    private var cachedContextTokens: [Int32] = []

    private var samplingParams = mercan_sampler_default_params()
    private var temporaryInvalidBytes: [CChar] = []
    private var generatedStopTail = ""
    /// The prompt itself opened a `<think>` block, so the model's output starts
    /// inside it. The tag is re-emitted with the first piece so UI-side
    /// ThinkTagStripper sees a complete <think>…</think> block.
    private var pendingThinkOpen = false

    private var entropyWindow: [Float] = []
    private let entropyWindowSize = 10
    var currentEntropy: Float = 0
    var averageEntropy: Float = 0

    private init(
        model: OpaquePointer,
        context: OpaquePointer,
        architectureName: String,
        tokenizerName: String
    ) {
        self.model = model
        self.context = context
        self.architectureName = architectureName
        self.tokenizerName = tokenizerName
        self.nLen = Int32(mercan_context_size(context))
    }

    deinit {
        mercan_context_free(context)
        mercan_model_free(model)
        mercan_backend_free()
    }

    static func create_context(
        path: String,
        contextSize: UInt32 = 4096,
        onProgress: (@Sendable (Float) -> Void)? = nil
    ) throws -> MercanRuntimeContext {
        mercan_backend_init()
        onProgress?(0)

        var modelParams = mercan_model_default_params()
        modelParams.n_gpu_layers = -1
        modelParams.use_mmap = true
        modelParams.check_tensors = false

        guard let model = mercan_model_load(path, modelParams) else {
            let detail = lastRuntimeError()
            mercan_backend_free()
            throw LlamaError.couldNotInitializeContext(path: path, detail: detail)
        }

        var contextParams = mercan_context_default_params()
        contextParams.n_ctx = contextSize
        contextParams.n_batch = min(UInt32(512), contextSize)

        #if targetEnvironment(simulator)
        let threadCount = max(1, min(4, ProcessInfo.processInfo.processorCount - 2))
        #else
        let threadCount = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        #endif
        contextParams.n_threads = Int32(threadCount)
        contextParams.n_threads_batch = Int32(threadCount)

        guard let context = mercan_context_create(model, contextParams) else {
            let detail = lastRuntimeError()
            mercan_model_free(model)
            mercan_backend_free()
            throw LlamaError.couldNotInitializeContext(path: path, detail: detail)
        }

        let architecture = mercan_model_architecture(model).map { String(cString: $0) } ?? "unknown"
        let tokenizer = mercan_model_tokenizer(model).map { String(cString: $0) } ?? "unknown"
        onProgress?(1)

        return MercanRuntimeContext(
            model: model,
            context: context,
            architectureName: architecture,
            tokenizerName: tokenizer
        )
    }

    private static func lastRuntimeError() -> String {
        guard let error = mercan_last_error() else { return "Unknown libmercan error." }
        let text = String(cString: error)
        return text.isEmpty ? "Unknown libmercan error." : text
    }

    private func runtimeError() -> String {
        Self.lastRuntimeError()
    }

    func setSampling(_ configuration: SamplingConfiguration) {
        var params = mercan_sampler_default_params()
        params.temperature = configuration.temperature
        params.top_k = configuration.topK
        params.top_p = configuration.topP
        params.min_p = configuration.minP
        params.repeat_penalty = configuration.repeatPenalty
        params.repeat_last_n = configuration.repeatLastN
        samplingParams = params
    }

    func supportsReasoningControl() -> Bool {
        architectureName == "lfm3"
    }

    func model_info() -> String {
        "Mercan \(architectureName) / \(tokenizerName)"
    }

    func completion_init(text: String) throws {
        let tokens = try tokenize(text: text, addSpecial: false, parseSpecial: true)
        try validatePrompt(tokens)

        if mercan_context_reset(context, false) != 0 {
            throw LlamaError.decodeFailed(runtimeError())
        }

        resetStreamingState()
        pendingThinkOpen = text.hasSuffix("<think>")
        try decode(tokens, startingAt: 0)
        cachedContextTokens = tokens
        nCur = Int32(tokens.count)
    }

    /// Reuses the longest common token prefix already resident in libmercan's KV
    /// cache, discards only the divergent tail, then decodes the new suffix.
    func completion_init_with_cache(text: String) throws {
        let tokens = try tokenize(text: text, addSpecial: false, parseSpecial: true)
        try validatePrompt(tokens)

        var commonPrefix = 0
        let limit = min(cachedContextTokens.count, tokens.count)
        while commonPrefix < limit && cachedContextTokens[commonPrefix] == tokens[commonPrefix] {
            commonPrefix += 1
        }

        if mercan_context_rewind(context, UInt32(commonPrefix)) != 0 {
            if mercan_context_reset(context, false) != 0 {
                throw LlamaError.decodeFailed(runtimeError())
            }
            commonPrefix = 0
        }

        resetStreamingState()
        pendingThinkOpen = text.hasSuffix("<think>")
        try decode(tokens, startingAt: commonPrefix)
        cachedContextTokens = tokens
        nCur = Int32(tokens.count)
    }

    private func validatePrompt(_ tokens: [Int32]) throws {
        guard !tokens.isEmpty else {
            throw LlamaError.tokenizationFailed("Prompt produced no tokens.")
        }
        guard tokens.count < Int(nLen) else {
            throw LlamaError.decodeFailed("Prompt is larger than the selected context window.")
        }
    }

    private func decode(_ tokens: [Int32], startingAt startIndex: Int) throws {
        guard startIndex <= tokens.count else {
            throw LlamaError.decodeFailed("Invalid KV prefix position.")
        }

        let chunkSize = 512
        var start = startIndex
        while start < tokens.count {
            let end = min(start + chunkSize, tokens.count)
            let slice = Array(tokens[start..<end])
            let rc = slice.withUnsafeBufferPointer { buffer -> Int32 in
                guard let base = buffer.baseAddress else { return -1 }
                return mercan_decode(context, base, Int32(buffer.count))
            }
            if rc != 0 {
                is_done = true
                throw LlamaError.decodeFailed(runtimeError())
            }
            start = end
        }
    }

    private func resetStreamingState() {
        is_done = false
        temporaryInvalidBytes.removeAll(keepingCapacity: true)
        generatedStopTail = ""
        entropyWindow.removeAll(keepingCapacity: true)
        currentEntropy = 0
        averageEntropy = 0
    }

    func completion_loop() throws -> String {
        let piece = try completionStep()
        if pendingThinkOpen {
            pendingThinkOpen = false
            return "<think>" + piece
        }
        return piece
    }

    private func completionStep() throws -> String {
        guard nCur < nLen else {
            is_done = true
            return ""
        }

        guard let logits = mercan_logits(context) else {
            is_done = true
            throw LlamaError.decodeFailed("libmercan returned no logits.")
        }

        let vocabSize = Int(mercan_vocab_size(model))
        guard vocabSize > 0 else {
            is_done = true
            throw LlamaError.decodeFailed("Model vocabulary is unavailable.")
        }

        var maxLogit = -Float.infinity
        for i in 0..<vocabSize {
            maxLogit = max(maxLogit, logits[i])
        }
        var sumExp: Float = 0
        for i in 0..<vocabSize {
            sumExp += exp(logits[i] - maxLogit)
        }
        let logSumExp = log(sumExp) + maxLogit
        var entropy: Float = 0
        if sumExp.isFinite && sumExp > 0 {
            for i in 0..<vocabSize {
                let p = exp(logits[i] - logSumExp)
                if p > 0 { entropy -= p * log2(p) }
            }
        }
        entropyWindow.append(entropy)
        if entropyWindow.count > entropyWindowSize {
            entropyWindow.removeFirst()
        }
        currentEntropy = entropy
        averageEntropy = entropyWindow.isEmpty ? 0 : entropyWindow.reduce(0, +) / Float(entropyWindow.count)

        let nextToken = mercan_sample_next(context, samplingParams)
        if nextToken < 0 {
            is_done = true
            throw LlamaError.samplingFailed(runtimeError())
        }

        let eos = mercan_eos_token(model)
        // Mercan SFT uses 32001 (<|im_end|>) as the normal assistant-message
        // terminator. EOS=2 remains the conversation/document boundary.
        // Stop on structural token IDs before converting them back to text.
        // Those IDs are NedoLM-specific: in other vocabularies (e.g. LFM3)
        // 32000/32001 are ordinary words, so other architectures use the
        // vocabulary's own end-of-generation set instead.
        let isStop = isNedoLM
            ? (nextToken == eos || nextToken == 32001 || nextToken == 32000)
            : (nextToken == eos || mercan_token_is_eog(model, nextToken))
        if isStop {
            is_done = true
            let tail = String(cString: temporaryInvalidBytes + [0])
            temporaryInvalidBytes.removeAll(keepingCapacity: true)
            return tail
        }

        let pieceBytes = tokenToPiece(nextToken)
        temporaryInvalidBytes.append(contentsOf: pieceBytes)

        let piece: String
        if let valid = String(validatingUTF8: temporaryInvalidBytes + [0]) {
            piece = valid
            temporaryInvalidBytes.removeAll(keepingCapacity: true)
        } else {
            piece = ""
        }

        generatedStopTail += piece
        if generatedStopTail.count > 64 {
            generatedStopTail = String(generatedStopTail.suffix(64))
        }

        if generatedStopTail.contains("<|im_end|>") || generatedStopTail.contains("<|endoftext|>") {
            is_done = true
            return piece
        }

        var tokenToDecode = nextToken
        let rc = withUnsafePointer(to: &tokenToDecode) { ptr in
            mercan_decode(context, ptr, 1)
        }
        if rc != 0 {
            is_done = true
            throw LlamaError.decodeFailed(runtimeError())
        }

        cachedContextTokens.append(nextToken)
        nCur += 1
        return piece
    }

    /// NedoLM/Mercan models use Turkish ChatML roles and fixed structural IDs.
    private var isNedoLM: Bool { architectureName == "nedolm" }

    /// Upstream ChatML roles used by non-NedoLM architectures (LFM3).
    private func upstreamChatRole(_ role: String) -> String {
        switch role.lowercased() {
        case "system", "sistem":
            return "system"
        case "user", "kullanici", "kullanıcı":
            return "user"
        case "assistant", "asistan":
            return "assistant"
        case "tool", "araç", "arac":
            return "tool"
        default:
            return role
        }
    }

    private func canonicalChatRole(_ role: String) -> String {
        switch role.lowercased() {
        case "system", "sistem":
            return "sistem"
        case "user", "kullanici", "kullanıcı":
            return "kullanici"
        case "assistant", "asistan":
            return "asistan"
        case "tool", "araç", "arac":
            return "araç"
        default:
            return role
        }
    }

    func apply_chat_template(
        messages: [(role: String, content: String)],
        enableThinking: Bool = false
    ) -> String {
        guard isNedoLM else {
            // LFM3 chat template: explicit BOS and English ChatML roles.
            // Reasoning on starts an open think block; off pre-fills a closed one.
            var result = "<|startoftext|>"
            for message in messages {
                let role = upstreamChatRole(message.role)
                result += "<|im_start|>\(role)\n\(message.content)<|im_end|>\n"
            }
            // Upstream LFM3 uses an open <think> for reasoning and an empty
            // <think></think> prefix for no-think mode. Some LFM weights may
            // still emit reasoning when off; that behavior is model-dependent.
            result += enableThinking
                ? "<|im_start|>assistant\n<think>"
                : "<|im_start|>assistant\n<think></think>"
            return result
        }

        var result = ""
        for message in messages {
            let role = canonicalChatRole(message.role)
            result += "<|im_start|>\(role)\n\(message.content)<|im_end|>\n"
        }
        result += "<|im_start|>asistan\n"
        return result
    }

    func countTokens(text: String) -> Int {
        (try? tokenize(text: text, addSpecial: false, parseSpecial: true).count) ?? 0
    }

    func clear() {
        _ = mercan_context_reset(context, false)
        cachedContextTokens.removeAll(keepingCapacity: true)
        nCur = 0
        is_done = true
        temporaryInvalidBytes.removeAll()
        generatedStopTail = ""
        entropyWindow.removeAll()
    }

    func clearGenerationState() {
        is_done = true
        temporaryInvalidBytes.removeAll(keepingCapacity: true)
        generatedStopTail = ""
    }

    private func tokenize(text: String, addSpecial: Bool, parseSpecial: Bool) throws -> [Int32] {
        let byteCount = text.utf8.count
        let required = text.withCString { ptr in
            mercan_tokenize(model, ptr, byteCount, addSpecial, parseSpecial, nil, 0)
        }

        if required == 0 {
            if text.isEmpty { return [] }
            throw LlamaError.tokenizationFailed(runtimeError())
        }

        let capacity = Int(required < 0 ? -required : required)
        guard capacity > 0 else { return [] }

        var tokens = [Int32](repeating: 0, count: capacity)
        let got = tokens.withUnsafeMutableBufferPointer { buffer in
            text.withCString { ptr in
                mercan_tokenize(
                    model,
                    ptr,
                    byteCount,
                    addSpecial,
                    parseSpecial,
                    buffer.baseAddress,
                    Int32(buffer.count)
                )
            }
        }

        if got < 0 {
            let needed = Int(-got)
            tokens = [Int32](repeating: 0, count: needed)
            let retry = tokens.withUnsafeMutableBufferPointer { buffer in
                text.withCString { ptr in
                    mercan_tokenize(
                        model,
                        ptr,
                        byteCount,
                        addSpecial,
                        parseSpecial,
                        buffer.baseAddress,
                        Int32(buffer.count)
                    )
                }
            }
            guard retry >= 0 else {
                throw LlamaError.tokenizationFailed(runtimeError())
            }
            tokens.removeSubrange(Int(retry)..<tokens.count)
            return tokens
        }

        if Int(got) < tokens.count {
            tokens.removeSubrange(Int(got)..<tokens.count)
        }
        return tokens
    }

    private func tokenToPiece(_ token: Int32) -> [CChar] {
        var small = [CChar](repeating: 0, count: 64)
        var n = small.withUnsafeMutableBufferPointer { buffer in
            mercan_token_to_piece(model, token, buffer.baseAddress, Int32(buffer.count), false)
        }

        if n >= 0 {
            return Array(small.prefix(Int(n)))
        }

        let needed = Int(-n)
        var large = [CChar](repeating: 0, count: needed)
        n = large.withUnsafeMutableBufferPointer { buffer in
            mercan_token_to_piece(model, token, buffer.baseAddress, Int32(buffer.count), false)
        }
        guard n >= 0 else { return [] }
        return Array(large.prefix(Int(n)))
    }
}
