import Foundation

actor MercanRuntimeEngine: InferenceEngine {
    private var runtimeContext: MercanRuntimeContext?
    private var samplingConfiguration = SamplingConfiguration.standard
    var isComplete: Bool = true

    init() {}

    func initialize(
        modelPath: String,
        contextSize: UInt32,
        onProgress: (@Sendable (Float) -> Void)? = nil
    ) async throws {
        let context = try await Task.detached(priority: .userInitiated) {
            try MercanRuntimeContext.create_context(
                path: modelPath,
                contextSize: contextSize,
                onProgress: onProgress
            )
        }.value
        runtimeContext = context
        await context.setSampling(samplingConfiguration)
    }

    func setSampling(_ configuration: SamplingConfiguration) async {
        samplingConfiguration = configuration
        if let runtimeContext {
            await runtimeContext.setSampling(configuration)
        }
    }

    func generateNext(messages: [(role: String, content: String)]) async throws {
        guard let context = runtimeContext else {
            throw NSError(
                domain: "MercanRuntimeEngine",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "libmercan runtime is not initialized"]
            )
        }

        await LiveTokenCoalescer.shared.beginGeneration()
        await context.setSampling(samplingConfiguration)
        let finalPrompt = await context.apply_chat_template(messages: messages)
        try await context.completion_init_with_cache(text: finalPrompt)
        isComplete = false
    }

    func encodePrompt(messages: [(role: String, content: String)]) async throws {
        guard let context = runtimeContext else {
            throw NSError(
                domain: "MercanRuntimeEngine",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "libmercan runtime is not initialized"]
            )
        }
        let finalPrompt = await context.apply_chat_template(messages: messages)
        try await context.completion_init(text: finalPrompt)
        isComplete = true
    }

    func streamToken() async throws -> String? {
        guard let context = runtimeContext else { return nil }

        if await context.is_done {
            isComplete = true
            await LiveTokenCoalescer.shared.finishGeneration()
            return nil
        }

        let token = try await context.completion_loop()
        if !token.isEmpty {
            await LiveTokenCoalescer.shared.ingest(token)
        }
        if await context.is_done {
            isComplete = true
            await LiveTokenCoalescer.shared.finishGeneration()
        }
        return token
    }

    func stop() async {
        isComplete = true
        await LiveTokenCoalescer.shared.finishGeneration()
        if let context = runtimeContext {
            await context.clearGenerationState()
        }
    }

    func resume() async {
        isComplete = false
    }

    func clear() async {
        if let context = runtimeContext {
            await context.clear()
        }
        await LiveTokenCoalescer.shared.clearPresentation()
        isComplete = true
    }

    func modelInfo() -> String {
        "Mercan Runtime (libmercan)"
    }

    func clearGenerationState() async {
        if let context = runtimeContext {
            await context.clearGenerationState()
        }
    }

    var currentEntropy: Float {
        get async { await runtimeContext?.currentEntropy ?? 0 }
    }

    var averageEntropy: Float {
        get async { await runtimeContext?.averageEntropy ?? 0 }
    }

    func countTokens(for messages: [(role: String, content: String)]) async -> Int {
        guard let context = runtimeContext else { return 0 }
        let prompt = await context.apply_chat_template(messages: messages)
        return await context.countTokens(text: prompt)
    }

    func deinitialize() async {
        runtimeContext = nil
        await LiveTokenCoalescer.shared.clearPresentation()
        isComplete = true
    }
}
