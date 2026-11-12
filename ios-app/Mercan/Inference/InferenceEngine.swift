import Foundation

struct SamplingConfiguration: Sendable {
    var temperature: Float
    var topK: Int32
    var topP: Float
    var minP: Float
    var repeatPenalty: Float
    var repeatLastN: Int32

    static let standard = SamplingConfiguration(
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        minP: 0.05,
        repeatPenalty: 1.10,
        repeatLastN: 64
    )
}

protocol InferenceEngine: Actor {
    var isComplete: Bool { get }

    func initialize(modelPath: String, contextSize: UInt32, onProgress: (@Sendable (Float) -> Void)?) async throws
    func setSampling(_ configuration: SamplingConfiguration) async
    func setReasoningEnabled(_ enabled: Bool) async
    func supportsReasoningControl() async -> Bool
    func generateNext(messages: [(role: String, content: String)]) async throws
    func encodePrompt(messages: [(role: String, content: String)]) async throws
    func streamToken() async throws -> String?
    func stop() async
    func resume() async
    func clear() async
    func clearGenerationState() async
    func modelInfo() -> String
    func deinitialize() async
    func countTokens(for messages: [(role: String, content: String)]) async -> Int
}
