import Foundation
import UIKit

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
    var rec: Bool?
    /// Official model release date — catalog entries must be within 12 months.
    var released: Date = .distantPast
}

struct ChatMessage: Identifiable, Codable {
    let id: UUID
    let content: String
    let isUser: Bool
    let timestamp: Date

    init(id: UUID = UUID(), content: String, isUser: Bool, timestamp: Date) {
        self.id = id
        self.content = content
        self.isUser = isUser
        self.timestamp = timestamp
    }

    static let videoTranscriptAttachmentPrefix = "Interview transcript attached"

    var isVideoTranscriptAttachment: Bool {
        isUser && content.hasPrefix(Self.videoTranscriptAttachmentPrefix)
    }

    var reasoningContent: String {
        isUser ? "" : ReasoningOutput.parse(content).reasoning
    }

    var displayContent: String {
        isUser ? content : ReasoningOutput.parse(content).answer
    }
}

/// Separates LFM's streamed <think> blocks from the user-facing answer.
/// Also tolerates incomplete blocks during streaming and stray closing tags in
/// models that do not consistently honor the no-thinking prompt.
struct ReasoningOutput {
    let reasoning: String
    let answer: String
    let isInsideReasoning: Bool

    static func parse(_ raw: String) -> ReasoningOutput {
        var remaining = raw[...]
        var answerParts: [String] = []
        var reasoningParts: [String] = []
        var isInsideReasoning = false

        while let open = remaining.range(of: "<think>") {
            answerParts.append(String(remaining[..<open.lowerBound]))
            let inside = remaining[open.upperBound...]
            if let close = inside.range(of: "</think>") {
                reasoningParts.append(String(inside[..<close.lowerBound]))
                remaining = inside[close.upperBound...]
            } else {
                reasoningParts.append(String(inside))
                remaining = remaining[remaining.endIndex...]
                isInsideReasoning = true
                break
            }
        }

        answerParts.append(String(remaining))
        let answer = answerParts.joined()
            .replacingOccurrences(of: "</think>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return ReasoningOutput(
            reasoning: reasoningParts.joined(separator: "\n\n")
                .trimmingCharacters(in: .whitespacesAndNewlines),
            answer: answer,
            isInsideReasoning: isInsideReasoning
        )
    }
}

// MARK: - Special Token Filter

class SpecialTokenFilter {
    private static let specialTokens: [String] = [
        "<|im_end|>", "<|im_start|>assistant", "<|im_start|>user",
        "<|im_start|>system", "<|im_start|>", "<|endoftext|>",
        "<|end_of_text|>", "</s>", "<s>", "[INST]", "[/INST]",
        "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]",
        "<end_of_turn>", "<start_of_turn>model", "<start_of_turn>user",
        "<start_of_turn>system", "<start_of_turn>", "<bos>",
        "<turn|>", "<|turn>model", "<|turn>user", "<|turn>system", "<|turn>",
        "<eos>", "<|tool_response>"
    ]

    private static let maxTokenLength: Int = {
        specialTokens.map(\.count).max() ?? 15
    }()

    private var buffer = ""

    func process(_ input: String) -> String {
        buffer += input
        var output = ""

        while !buffer.isEmpty {
            // Check if buffer matches any special token exactly
            var matched = false
            for token in Self.specialTokens {
                if buffer.hasPrefix(token) {
                    // Remove matched special token
                    buffer.removeFirst(token.count)
                    matched = true
                    break
                }
            }
            if matched { continue }

            // Check if buffer could be the start of a special token
            let couldBePrefix = Self.specialTokens.contains { token in
                token.hasPrefix(buffer) && buffer.count < token.count
            }

            if couldBePrefix {
                // Hold back — might be partial special token
                break
            }

            // Safe to emit the first character
            output.append(buffer.removeFirst())
        }

        return output
    }

    func flush() -> String {
        // On completion, emit whatever is left — it wasn't a special token
        let remaining = buffer
        buffer = ""
        // But still strip any complete special tokens in the remaining
        var result = remaining
        for token in Self.specialTokens {
            result = result.replacingOccurrences(of: token, with: "")
        }
        return result
    }

    func reset() {
        buffer = ""
    }
}

// MARK: - Think Tag Stripper

class ThinkTagStripper {
    private enum State {
        case lookingForOpen   // Haven't seen <think> yet
        case insideThink      // Between <think> and </think>
        case passThrough      // After </think>, emit everything
    }

    private var state: State = .lookingForOpen
    private var buffer = ""

    var isInsideThink: Bool {
        state == .insideThink || state == .lookingForOpen
    }

    func process(_ input: String) -> String {
        buffer += input

        guard !buffer.isEmpty else { return "" }

        switch state {
        case .lookingForOpen:
            // Check if buffer contains <think>
            if let range = buffer.range(of: "<think>") {
                let before = String(buffer[buffer.startIndex..<range.lowerBound])
                buffer = String(buffer[range.upperBound...])
                state = .insideThink
                return before
            }
            // Check if any suffix of buffer is a prefix of "<think>"
            let holdBack = suffixPrefixOverlap(buffer, "<think>")
            if holdBack > 0 {
                let safe = String(buffer.dropLast(holdBack))
                buffer = String(buffer.suffix(holdBack))
                return safe
            }
            // No chance of <think>, pass through
            let out = buffer
            buffer = ""
            state = .passThrough
            return out

        case .insideThink:
            // Look for </think>
            if let range = buffer.range(of: "</think>") {
                buffer = String(buffer[range.upperBound...])
                state = .passThrough
                let out = buffer
                buffer = ""
                return out
            }
            // Check if any suffix could be start of "</think>"
            let hold = suffixPrefixOverlap(buffer, "</think>")
            if hold > 0 {
                buffer = String(buffer.suffix(hold))
            } else {
                buffer = ""
            }
            return ""

        case .passThrough:
            let out = buffer
            buffer = ""
            return out
        }
    }

    func flush() -> String {
        switch state {
        case .lookingForOpen:
            // Never found <think>, emit everything
            let out = buffer
            buffer = ""
            return out
        case .insideThink:
            // Unclosed think block — discard
            buffer = ""
            return ""
        case .passThrough:
            let out = buffer
            buffer = ""
            return out
        }
    }

    func reset() {
        state = .lookingForOpen
        buffer = ""
    }

    /// Returns the length of the longest suffix of `text` that is a prefix of `tag`.
    private func suffixPrefixOverlap(_ text: String, _ tag: String) -> Int {
        let maxCheck = min(text.count, tag.count - 1)
        guard maxCheck > 0 else { return 0 }
        for len in stride(from: maxCheck, through: 1, by: -1) {
            if tag.hasPrefix(String(text.suffix(len))) {
                return len
            }
        }
        return 0
    }
}

struct Conversation: Identifiable, Codable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    var transcript: String?
    /// Relative path under Documents (e.g. `transcripts/<id>.txt`) for long transcripts.
    var transcriptFilename: String?
    let createdAt: Date
    var updatedAt: Date

    init(
        id: UUID = UUID(),
        title: String = "New Chat",
        messages: [ChatMessage] = [],
        transcript: String? = nil,
        transcriptFilename: String? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.transcript = transcript
        self.transcriptFilename = transcriptFilename
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    var displayTitle: String {
        if title != "New Chat" && !title.isEmpty {
            return title
        }
        if let firstUserMessage = messages.first(where: { $0.isUser }) {
            let content = firstUserMessage.content
            return content.count > 40 ? String(content.prefix(40)) + "..." : content
        }
        return title
    }

    var relativeDate: String {
        let calendar = Calendar.current
        if calendar.isDateInToday(updatedAt) { return String(localized: "Today") }
        if calendar.isDateInYesterday(updatedAt) { return String(localized: "Yesterday") }
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        return formatter.string(from: updatedAt)
    }
}

@MainActor
class LlamaState: ObservableObject {

    @Published var messages: [ChatMessage] = []
    @Published var currentResponse = ""
    @Published var isGenerating = false
    @Published var systemPrompt: String {
        didSet { UserDefaults.standard.set(systemPrompt, forKey: "systemPrompt") }
    }
    @Published var reasoningEnabled: Bool {
        didSet { UserDefaults.standard.set(reasoningEnabled, forKey: "reasoningEnabled") }
    }
    @Published var modelSupportsReasoning = false
    @Published var webSearchEnabled: Bool {
        didSet { UserDefaults.standard.set(webSearchEnabled, forKey: "webSearchEnabled") }
    }
    @Published var cacheCleared = false
    @Published var contextTruncated = false
    @Published var isThinking = false
    @Published var modelConfidence: Float = 1.0
    @Published var contextSize: UInt32 {
        didSet {
            UserDefaults.standard.set(Int(contextSize), forKey: "contextSize")
            reloadCurrentModel()
        }
    }
    @Published var isLoadingModel = false
    @Published var modelLoadProgress = 0.0
    @Published var currentModelName = ""
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    @Published var currentConversation: Conversation?
    @Published var conversationTranscript: String?
    @Published private(set) var transcriptCharacterCount = 0
    @Published var attachedVideoThumbnail: UIImage?
    @Published var attachedTranscriptionJobId: UUID?
    @Published var modelSuspendedForSpeech = false
    @Published var modelLoadError: String?
    @Published var speakRepliesEnabled: Bool {
        didSet { UserDefaults.standard.set(speakRepliesEnabled, forKey: "speakRepliesEnabled") }
    }

    @Published var temperature: Double = 0.7 {
        didSet { UserDefaults.standard.set(temperature, forKey: "temperature") }
    }
    @Published var topP: Double = 0.95 {
        didSet { UserDefaults.standard.set(topP, forKey: "topP") }
    }
    @Published var topK: Int = 40 {
        didSet { UserDefaults.standard.set(topK, forKey: "topK") }
    }
    @Published var minP: Double = 0.05 {
        didSet { UserDefaults.standard.set(minP, forKey: "minP") }
    }
    @Published var repeatPenalty: Double = 1.10 {
        didSet { UserDefaults.standard.set(repeatPenalty, forKey: "repeatPenalty") }
    }

    @Published var contextTokenCount: Int = 0
    @Published var lastPromptTokenCount: Int = 0
    @Published var generatedTokenCount: Int = 0
    @Published var lastGenerationDuration: TimeInterval = 0
    @Published var lastGenerationTokensPerSecond: Double = 0
    @Published var attachedDocument: AttachedDocument?
    @Published var documentImportError: String?
    @Published var benchmarkResult: BenchmarkResult?
    @Published var isBenchmarking = false

    private var generationStartedAt: Date?

    private var samplingConfiguration: SamplingConfiguration {
        SamplingConfiguration(
            temperature: Float(temperature),
            topK: Int32(topK),
            topP: Float(topP),
            minP: Float(minP),
            repeatPenalty: Float(repeatPenalty),
            repeatLastN: 64
        )
    }

    let speechSynthesizer = SpeechSynthesizerService()
    private let webSearchService = WebSearchService()
    private var suspendedModelURL: URL?

    // Whisper models (reused download system)
    let whisperModels: [Model] = [
        Model(name: "Whisper Small EN (q5_1)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin", filename: "ggml-small.en-q5_1.bin"),
        Model(name: "Whisper Base EN", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin", filename: "ggml-base.en.bin"),
    ]
    @Published var downloadedWhisperModels: [Model] = []

    func defaultWhisperModelPath() -> String? {
        let preferred = "ggml-small.en-q5_1.bin"
        let docsURL = getDocumentsDirectory().appendingPathComponent(preferred)
        
        // If already in documents, use it
        if FileManager.default.fileExists(atPath: docsURL.path) {
            return docsURL.path
        }
        
        // Check if bundled in app (from Silo/models/whisper/)
        if let bundleURL = Bundle.main.url(forResource: "ggml-small.en-q5_1", withExtension: "bin", subdirectory: "models/whisper") {
            // Copy to documents for runtime use
            do {
                try FileManager.default.copyItem(at: bundleURL, to: docsURL)
                print("Copied bundled whisper model to documents")
                return docsURL.path
            } catch {
                print("Failed to copy bundled model: \(error)")
                return bundleURL.path  // fallback to bundle path directly
            }
        }
        
        // fallback to first downloaded whisper in docs
        if let first = downloadedWhisperModels.first {
            let url = getDocumentsDirectory().appendingPathComponent(first.filename)
            if FileManager.default.fileExists(atPath: url.path) {
                return url.path
            }
        }
        return nil
    }

    func downloadWhisperModel(_ model: Model) {
        startDownload(modelName: model.name, modelUrl: model.url, filename: model.filename)
        // After download, refresh list (simplified; in real would observe)
        Task {
            try? await Task.sleep(for: .seconds(1))
            await MainActor.run { self.loadWhisperModelsFromDisk() }
        }
    }

    // Active download tracking — persists across modal dismiss/re-present
    @Published var activeDownloads: [String: Double] = [:]  // filename -> progress
    var downloadTasks: [String: URLSessionDownloadTask] = [:]
    var downloadObservations: [String: NSKeyValueObservation] = [:]

    // Pending download metadata — maps filename to (modelName, modelUrl) for background session reconnection
    private var pendingDownloadMeta: [String: (name: String, url: String)] = [:]

    func startDownload(modelName: String, modelUrl: String, filename: String) {
        guard let url = URL(string: modelUrl) else { return }
        let fileURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)

        // Remove any leftover partial file
        try? FileManager.default.removeItem(at: fileURL)

        activeDownloads[filename] = 0.0
        pendingDownloadMeta[filename] = (name: modelName, url: modelUrl)

        let task = URLSession.shared.downloadTask(with: url) { [weak self] temporaryURL, response, error in
            if let error = error as? NSError, error.code == NSURLErrorCancelled {
                Task { @MainActor [weak self] in
                    self?.cleanupDownload(filename: filename)
                }
                return
            }

            if let error = error {
                print("Download error: \(error.localizedDescription)")
                Task { @MainActor [weak self] in
                    self?.cleanupDownload(filename: filename)
                }
                return
            }

            guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode) else {
                print("Server error")
                Task { @MainActor [weak self] in
                    self?.cleanupDownload(filename: filename)
                }
                return
            }

            do {
                if let temporaryURL = temporaryURL {
                    try FileManager.default.copyItem(at: temporaryURL, to: fileURL)

                    // Validate file is at least 1MB (a real GGUF is always bigger)
                    let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
                    let size = attrs[.size] as? Int64 ?? 0
                    if size < 1_000_000 {
                        try FileManager.default.removeItem(at: fileURL)
                        print("Downloaded file too small, likely corrupt — removed")
                        Task { @MainActor [weak self] in
                            self?.cleanupDownload(filename: filename)
                        }
                        return
                    }

                    print("Writing to \(filename) completed (\(size) bytes)")

                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.cleanupDownload(filename: filename)
                        self.cacheCleared = false
                        self.registerDownloadedModel(filename: filename, fallbackName: modelName, url: modelUrl)
                    }
                }
            } catch {
                print("File error: \(error.localizedDescription)")
                try? FileManager.default.removeItem(at: fileURL)
                Task { @MainActor [weak self] in
                    self?.cleanupDownload(filename: filename)
                }
            }
        }

        let observation = task.progress.observe(\.fractionCompleted) { [weak self] progress, _ in
            Task { @MainActor [weak self] in
                self?.activeDownloads[filename] = progress.fractionCompleted
            }
        }

        downloadTasks[filename] = task
        downloadObservations[filename] = observation
        task.resume()
    }

    func cancelDownload(filename: String) {
        downloadTasks[filename]?.cancel()
        cleanupDownload(filename: filename)
    }

    private func cleanupDownload(filename: String) {
        downloadTasks.removeValue(forKey: filename)
        downloadObservations.removeValue(forKey: filename)
        activeDownloads.removeValue(forKey: filename)
        pendingDownloadMeta.removeValue(forKey: filename)
    }

    private var inferenceEngine: InferenceEngine?
    var conversationManager: ConversationManager?

    var isModelLoaded: Bool { inferenceEngine != nil }

    private func modelFilesInDocuments() -> [URL] {
        let documents = getDocumentsDirectory()
        let files = (try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: nil)) ?? []
        return files.filter { url in
            let ext = url.pathExtension.lowercased()
            if ext == "mercan" { return true }
            // Compatibility with the previous iOS build, which downloaded the
            // Mercan v1 GGUF container using a .gguf filename.
            return ext == "gguf" && url.lastPathComponent == "Mercan-0.8B-SFT.gguf"
        }
    }

    private func fileSize(at url: URL) -> Int64 {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
    }

    private func resolveModelFileURL() -> URL? {
        let models = modelFilesInDocuments()
        guard !models.isEmpty else { return nil }

        #if targetEnvironment(simulator)
        let loadable = models.filter { fileSize(at: $0) <= Self.simulatorMaxModelBytes }
        if let smallest = loadable.min(by: { fileSize(at: $0) < fileSize(at: $1) }) {
            return smallest
        }
        return nil
        #else

        if !currentModelName.isEmpty,
           let match = models.first(where: { $0.deletingPathExtension().lastPathComponent == currentModelName }) {
            return match
        }
        return models.first
        #endif
    }

    #if targetEnvironment(simulator)
    private func bootstrapSimulatorModelIfNeeded() async {
        if resolveModelFileURL() != nil {
            if !isModelLoaded {
                _ = await ensureModelLoaded()
            }
            return
        }
        modelLoadError = String(localized: "No loadable Mercan model found. Downloading the default Mercan model…")
        let model = Self.defaultModel
        if !FileManager.default.fileExists(atPath: getDocumentsDirectory().appendingPathComponent(model.filename).path) {
            downloadModel(model)
        }
    }

    func downloadModel(_ model: Model) {
        guard let url = URL(string: model.url) else { return }
        let filename = model.filename
        guard activeDownloads[filename] == nil else { return }

        let fileURL = getDocumentsDirectory().appendingPathComponent(filename)
        try? FileManager.default.removeItem(at: fileURL)

        activeDownloads[filename] = 0.0
        pendingDownloadMeta[filename] = (name: model.name, url: model.url)

        let task = URLSession.shared.downloadTask(with: url) { [weak self] temporaryURL, response, error in
            guard let self else { return }
            if let error = error as? NSError, error.code == NSURLErrorCancelled {
                Task { @MainActor in self.cleanupDownload(filename: filename) }
                return
            }
            if error != nil {
                Task { @MainActor in self.cleanupDownload(filename: filename) }
                return
            }
            guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode),
                  let temporaryURL = temporaryURL else {
                Task { @MainActor in self.cleanupDownload(filename: filename) }
                return
            }
            do {
                try FileManager.default.copyItem(at: temporaryURL, to: fileURL)
                let size = (try? FileManager.default.attributesOfItem(atPath: fileURL.path)[.size] as? Int64) ?? 0
                if size < 1_000_000 {
                    try? FileManager.default.removeItem(at: fileURL)
                    Task { @MainActor in self.cleanupDownload(filename: filename) }
                    return
                }
                Task { @MainActor in
                    self.cleanupDownload(filename: filename)
                    self.registerDownloadedModel(filename: filename, fallbackName: model.name, url: model.url)
                    try? await self.loadModel(modelUrl: fileURL)
                }
            } catch {
                Task { @MainActor in self.cleanupDownload(filename: filename) }
            }
        }
        downloadTasks[filename] = task
        downloadObservations[filename] = task.progress.observe(\.fractionCompleted) { [weak self] progress, _ in
            Task { @MainActor in self?.activeDownloads[filename] = progress.fractionCompleted }
        }
        task.resume()
    }
    #endif

    private func initializeEngine(at modelURL: URL) async throws {
        let engine = MercanRuntimeEngine()
        inferenceEngine = engine
        do {
            try await engine.initialize(modelPath: modelURL.path(), contextSize: contextSize) { [weak self] progress in
                Task { @MainActor in
                    self?.modelLoadProgress = Double(progress)
                }
            }
            await engine.setSampling(samplingConfiguration)
            await engine.setReasoningEnabled(reasoningEnabled)
            modelSupportsReasoning = await engine.supportsReasoningControl()
        } catch {
            modelSupportsReasoning = false
            inferenceEngine = nil
            throw error
        }
        currentModelName = modelURL.deletingPathExtension().lastPathComponent
        registerDownloadedModel(
            filename: modelURL.lastPathComponent,
            fallbackName: currentModelName
        )
    }

    /// Loads the on-disk Mercan model if the runtime was unloaded (e.g. after video transcription).
    func ensureModelLoaded() async -> Bool {
        if inferenceEngine != nil { return true }

        if modelSuspendedForSpeech {
            await resumeModelAfterSpeech()
            if inferenceEngine != nil { return true }
        }

        guard let modelURL = resolveModelFileURL() else {
            #if targetEnvironment(simulator)
            modelLoadError = String(localized: "No Mercan model found. Open Manage Models or wait for the automatic download.")
            Task { await bootstrapSimulatorModelIfNeeded() }
            #else
            modelLoadError = String(localized: "No model file found on device.")
            #endif
            return false
        }

        isLoadingModel = true
        modelLoadProgress = 0
        modelLoadError = nil
        defer { isLoadingModel = false }

        do {
            try await initializeEngine(at: modelURL)
            return true
        } catch {
            modelLoadError = Self.describeLoadError(error)
            return false
        }
    }

    // Streaming state promoted to instance vars for stop() access
    private var isStopped = false
    private var rawResponse = ""
    private var rawResponseParts: [String] = []
    private var displayParts: [String] = []
    private var displayPartsSinceLastFlush = 0
    private var tokenFilter = SpecialTokenFilter()
    private var thinkStripper = ThinkTagStripper()

    init() {
        self.systemPrompt = UserDefaults.standard.string(forKey: "systemPrompt") ?? ""
        self.reasoningEnabled = UserDefaults.standard.bool(forKey: "reasoningEnabled")
        self.webSearchEnabled = UserDefaults.standard.object(forKey: "webSearchEnabled") == nil
            ? true
            : UserDefaults.standard.bool(forKey: "webSearchEnabled")
        self.speakRepliesEnabled = UserDefaults.standard.bool(forKey: "speakRepliesEnabled")
        if UserDefaults.standard.object(forKey: "temperature") != nil {
            self.temperature = UserDefaults.standard.double(forKey: "temperature")
        }
        if UserDefaults.standard.object(forKey: "topP") != nil {
            self.topP = UserDefaults.standard.double(forKey: "topP")
        }
        if UserDefaults.standard.object(forKey: "topK") != nil {
            self.topK = max(1, UserDefaults.standard.integer(forKey: "topK"))
        }
        if UserDefaults.standard.object(forKey: "minP") != nil {
            self.minP = UserDefaults.standard.double(forKey: "minP")
        }
        if UserDefaults.standard.object(forKey: "repeatPenalty") != nil {
            self.repeatPenalty = UserDefaults.standard.double(forKey: "repeatPenalty")
        }
        let saved = UserDefaults.standard.integer(forKey: "contextSize")
        #if targetEnvironment(simulator)
        self.contextSize = saved > 0 ? UInt32(min(saved, 2048)) : 2048
        #else
        self.contextSize = saved > 0 ? UInt32(saved) : 4096
        #endif
        loadModelsFromDisk()
        loadDownloadableModels()
        loadWhisperModelsFromDisk()
        if downloadedModels.isEmpty {
            downloadDefaultModel()
        } else {
            #if targetEnvironment(simulator)
            Task { await bootstrapSimulatorModelIfNeeded() }
            #endif
        }
    }

    #if targetEnvironment(simulator)
    /// GGUF larger than this usually fails to load in Simulator (use LFM2.5 instead).
    private static let simulatorMaxModelBytes: Int64 = 1_600_000_000
    #endif

    private func catalogModel(forFilename filename: String) -> Model? {
        downloadableModels.first { $0.filename == filename }
    }

    private func registerDownloadedModel(filename: String, fallbackName: String, url: String = "") {
        guard !downloadedModels.contains(where: { $0.filename == filename }) else { return }
        if let catalog = catalogModel(forFilename: filename) {
            var entry = catalog
            entry.status = "downloaded"
            downloadedModels.append(entry)
        } else {
            downloadedModels.append(Model(name: fallbackName, url: url, filename: filename, status: "downloaded"))
        }
        undownloadedModels.removeAll { $0.filename == filename }
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let allURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            let modelURLs = allURLs.filter { url in
                let ext = url.pathExtension.lowercased()
                return ext == "mercan" || (ext == "gguf" && url.lastPathComponent == "Mercan-0.8B-SFT.gguf")
            }
            for modelURL in modelURLs {
                // Skip files under 1MB — likely corrupt or partial
                let attrs = try? FileManager.default.attributesOfItem(atPath: modelURL.path)
                let size = attrs?[.size] as? Int64 ?? 0
                if size < 1_000_000 {
                    try? FileManager.default.removeItem(at: modelURL)
                    print("Removed corrupt/partial model: \(modelURL.lastPathComponent) (\(size) bytes)")
                    continue
                }
                let fallbackName = modelURL.deletingPathExtension().lastPathComponent
                registerDownloadedModel(filename: modelURL.lastPathComponent, fallbackName: fallbackName)
            }

            if let url = resolveModelFileURL() {
                Task {
                    try? await loadModel(modelUrl: url)
                }
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadWhisperModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let allURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            let whisperURLs = allURLs.filter { $0.lastPathComponent.hasPrefix("ggml-") && $0.pathExtension.lowercased() == "bin" }
            for url in whisperURLs {
                let filename = url.lastPathComponent
                if !downloadedWhisperModels.contains(where: { $0.filename == filename }) {
                    if let catalog = whisperModels.first(where: { $0.filename == filename }) {
                        var entry = catalog
                        entry.status = "downloaded"
                        downloadedWhisperModels.append(entry)
                    } else {
                        downloadedWhisperModels.append(Model(name: filename, url: "", filename: filename, status: "downloaded"))
                    }
                }
            }
        } catch {
            print("Error loading whisper models from disk: \(error)")
        }
    }

    private func loadDownloadableModels() {
        for model in downloadableModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {
                registerDownloadedModel(filename: model.filename, fallbackName: model.name, url: model.url)
            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModel.rec = canRunModel(model.name)
                if !undownloadedModels.contains(where: { $0.filename == model.filename }) {
                    undownloadedModels.append(undownloadedModel)
                }
            }
        }
    }

    @Published var isDownloadingDefault = false
    @Published var defaultDownloadProgress = 0.0
    private var defaultDownloadObservation: NSKeyValueObservation?

    private func downloadDefaultModel() {
        let model = Self.defaultModelForPlatform
        guard let url = URL(string: model.url) else { return }
        let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)

        isDownloadingDefault = true

        let task = URLSession.shared.downloadTask(with: url) { [weak self] temporaryURL, response, error in
            if let error = error {
                print("Default model download error: \(error.localizedDescription)")
                Task { @MainActor in self?.finishDefaultDownload() }
                return
            }

            guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode) else {
                print("Default model download server error")
                Task { @MainActor in self?.finishDefaultDownload() }
                return
            }

            do {
                if let temporaryURL = temporaryURL {
                    try FileManager.default.copyItem(at: temporaryURL, to: fileURL)

                    let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
                    let size = attrs[.size] as? Int64 ?? 0
                    if size < 1_000_000 {
                        try FileManager.default.removeItem(at: fileURL)
                        print("Default model download too small, likely corrupt — removed")
                        Task { @MainActor in self?.finishDefaultDownload() }
                        return
                    }

                    print("Default model downloaded: \(model.filename) (\(size) bytes)")

                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.finishDefaultDownload()
                        self.registerDownloadedModel(filename: model.filename, fallbackName: model.name, url: model.url)
                        try? await self.loadModel(modelUrl: fileURL)
                    }
                }
            } catch {
                print("Default model save error: \(error.localizedDescription)")
                Task { @MainActor in self?.finishDefaultDownload() }
            }
        }

        defaultDownloadObservation = task.progress.observe(\.fractionCompleted) { [weak self] progress, _ in
            Task { @MainActor in
                self?.defaultDownloadProgress = progress.fractionCompleted
            }
        }

        task.resume()
    }

    private func finishDefaultDownload() {
        isDownloadingDefault = false
        defaultDownloadProgress = 0.0
        defaultDownloadObservation = nil
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }

    private static let catalogMaxAgeMonths = 12

    private static func catalogDate(_ year: Int, _ month: Int, _ day: Int) -> Date {
        Calendar.current.date(from: DateComponents(year: year, month: month, day: day)) ?? .distantPast
    }

    private static var catalogAgeCutoff: Date {
        Calendar.current.date(byAdding: .month, value: -catalogMaxAgeMonths, to: Date()) ?? .distantPast
    }

    private static func catalogEligible(_ models: [Model]) -> [Model] {
        models.filter { $0.released >= catalogAgeCutoff }
    }

    static let defaultModel = Model(
        name: "Mercan 0.8B SFT Q4_K_M (482 MiB)",
        url: "https://huggingface.co/MercanAI/Mercan-0.8B-SFT/resolve/main/model.mercan?download=true",
        filename: "Mercan-0.8B-SFT.mercan",
        status: "download",
        released: catalogDate(2026, 9, 11))

    static var defaultModelForPlatform: Model {
        defaultModel
    }

    // libmercan v0.1.2 intentionally accepts Mercan Format v1 artifacts.
    // Generic third-party GGUF entries stay out of the iOS catalog until they
    // are exported as .mercan or supported through a Mercan architecture provider.
    private static let allDownloadableModels: [Model] = [
        LlamaState.defaultModel
    ]

    private var downloadableModels: [Model] {
        Self.catalogEligible(Self.allDownloadableModels)
    }

    private func reloadCurrentModel() {
        guard !currentModelName.isEmpty else { return }
        let documentsURL = getDocumentsDirectory()
        let allFiles = (try? FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil)) ?? []
        if let modelURL = allFiles.first(where: { $0.deletingPathExtension().lastPathComponent == currentModelName }) {
            Task { try? await loadModel(modelUrl: modelURL) }
        }
    }

    func loadModel(modelUrl: URL) async throws {
        if isGenerating {
            await stop()
        }
        saveCurrentConversation()

        isLoadingModel = true
        modelLoadProgress = 0.0
        modelLoadError = nil

        await inferenceEngine?.deinitialize()
        inferenceEngine = nil
        modelSupportsReasoning = false

        do {
            try await initializeEngine(at: modelUrl)
        } catch {
            isLoadingModel = false
            modelLoadError = Self.describeLoadError(error)
            throw error
        }

        isLoadingModel = false
        modelLoadError = nil
        print("Loaded model")
    }

    private static func describeLoadError(_ error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }

    private static func transcriptSystemMessage(_ transcript: String) -> String {
        let maxChars = 100_000
        let body = transcript.count > maxChars ? String(transcript.prefix(maxChars)) + "\n…[truncated]" : transcript
        return """
        The user attached a video transcript. Answer only using information from this transcript. If the answer is not in the transcript, say so clearly.

        Transcript:
        \(body)
        """
    }

    private static func documentSystemMessage(_ document: AttachedDocument, query: String) -> String {
        let body = DocumentContextService.groundingText(for: document, query: query)
        return """
        The user attached a document named "\(document.name)". Use the selected local excerpts below as grounding context when relevant. If the answer is not present, say that clearly.

        Document excerpts:
        \(body)
        """
    }

    func resolvedTranscriptText() -> String? {
        if let conversationTranscript, !conversationTranscript.isEmpty {
            return conversationTranscript
        }
        if let inline = currentConversation?.transcript, !inline.isEmpty {
            return inline
        }
        return nil
    }

    /// Loads transcript text off the main thread when it is only stored on disk.
    func loadResolvedTranscriptText() async -> String? {
        if let cached = resolvedTranscriptText() {
            return cached
        }
        guard let filename = currentConversation?.transcriptFilename else { return nil }
        let url = getDocumentsDirectory().appendingPathComponent(filename)
        let text = await Task.detached(priority: .userInitiated) {
            try? String(contentsOf: url, encoding: .utf8)
        }.value
        if let text, !text.isEmpty {
            conversationTranscript = text
            transcriptCharacterCount = text.count
        }
        return text
    }

    private func setTranscriptContext(text: String?, jobId: UUID?) {
        conversationTranscript = text
        transcriptCharacterCount = text?.count ?? 0
        attachedTranscriptionJobId = jobId
        refreshAttachedVideoThumbnail()
    }

    func refreshAttachedVideoThumbnail() {
        if let jobId = attachedTranscriptionJobId,
           let image = VideoThumbnailGenerator.loadJobThumbnail(jobId: jobId) {
            attachedVideoThumbnail = image
            return
        }
        if let conversationId = currentConversation?.id,
           let image = VideoThumbnailGenerator.loadConversationThumbnail(conversationId: conversationId) {
            attachedVideoThumbnail = image
            return
        }
        attachedVideoThumbnail = nil
    }

    private func appendTranscriptContext(to chatMessages: inout [(role: String, content: String)]) async {
        guard let transcript = await loadResolvedTranscriptText(), !transcript.isEmpty else { return }
        chatMessages.append((role: "system", content: Self.transcriptSystemMessage(transcript)))
    }

    private func appendDocumentContext(
        to chatMessages: inout [(role: String, content: String)],
        query: String
    ) {
        guard let attachedDocument else { return }
        chatMessages.append((
            role: "system",
            content: Self.documentSystemMessage(attachedDocument, query: query)
        ))
    }

    private func saveTranscriptToDisk(conversationId: UUID, transcript: String) throws -> String {
        let dir = getDocumentsDirectory().appendingPathComponent("transcripts", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let relative = "transcripts/\(conversationId.uuidString).txt"
        let url = getDocumentsDirectory().appendingPathComponent(relative)
        try transcript.write(to: url, atomically: true, encoding: .utf8)
        return relative
    }

    func suspendModelForSpeech() async {
        guard inferenceEngine != nil else { return }
        if isGenerating { await stop() }

        suspendedModelURL = resolveModelFileURL()

        await inferenceEngine?.deinitialize()
        inferenceEngine = nil
        modelSuspendedForSpeech = true
    }

    func resumeModelAfterSpeech() async {
        guard modelSuspendedForSpeech else { return }
        defer {
            modelSuspendedForSpeech = false
            suspendedModelURL = nil
        }
        guard let modelURL = suspendedModelURL ?? resolveModelFileURL() else {
            modelLoadError = String(localized: "Could not find model file to reload.")
            return
        }

        isLoadingModel = true
        modelLoadProgress = 0
        modelLoadError = nil

        do {
            try await initializeEngine(at: modelURL)
        } catch {
            modelLoadError = Self.describeLoadError(error)
        }
        isLoadingModel = false
    }

    private func chatMessagesForInference() async -> [(role: String, content: String)] {
        var chatMessages: [(role: String, content: String)] = []
        if webSearchEnabled {
            chatMessages.append((
                role: "system",
                content: WebSearchService.augmentedSystemPrompt(base: systemPrompt)
            ))
        } else if !systemPrompt.isEmpty {
            chatMessages.append((role: "system", content: systemPrompt))
        }
        await appendTranscriptContext(to: &chatMessages)
        let latestQuery = messages.last(where: { $0.isUser })?.content ?? ""
        appendDocumentContext(to: &chatMessages, query: latestQuery)
        for msg in messages {
            chatMessages.append((role: msg.isUser ? "user" : "assistant",
                                 content: msg.isUser ? msg.content : msg.displayContent))
        }
        return chatMessages
    }

    /// Starts a new chat with an on-device video transcript as grounding context.
    func attachVideoTranscript(_ transcript: String, transcriptJobId: UUID? = nil) async {
        saveCurrentConversation()
        messages = []

        var text = transcript
        if let jobId = transcriptJobId {
            let jobTranscript = await Task.detached(priority: .userInitiated) {
                TranscriptionCheckpointStore.readTranscript(jobId: jobId)
            }.value
            if let jobTranscript, !jobTranscript.isEmpty {
                text = jobTranscript
            }
        }

        var conversation = conversationManager?.createNew() ?? Conversation()
        conversation.title = "Video chat"

        if let jobId = transcriptJobId {
            VideoThumbnailGenerator.copyJobThumbnailToConversation(jobId: jobId, conversationId: conversation.id)
        }

        if let relative = try? saveTranscriptToDisk(conversationId: conversation.id, transcript: text) {
            conversation.transcriptFilename = relative
            conversation.transcript = nil
        } else {
            conversation.transcript = text
        }
        currentConversation = conversation
        setTranscriptContext(text: text, jobId: transcriptJobId)

        let preview = "\(ChatMessage.videoTranscriptAttachmentPrefix) (\(text.count) characters). Ask questions about the video."
        messages = [ChatMessage(content: preview, isUser: true, timestamp: Date())]
        conversation.messages = messages
        conversationManager?.save(conversation)
        conversationManager?.currentConversationId = conversation.id
    }

    /// Removes video transcript context (e.g. when the user dismisses the banner).
    func clearVideoTranscriptAttachment() {
        setTranscriptContext(text: nil, jobId: nil)
        guard var conversation = currentConversation else { return }
        conversation.transcript = nil
        conversation.transcriptFilename = nil
        if messages.count == 1,
           let only = messages.first,
           only.isUser,
           only.isVideoTranscriptAttachment {
            messages = []
            conversation.messages = []
        }
        currentConversation = conversation
        conversationManager?.save(conversation)
    }

    func restoreToUndownloaded(filename: String) {
        // Check if this model is from the predefined downloadable models
        if let model = downloadableModels.first(where: { $0.filename == filename }) {
            // Only add back if not already in undownloadedModels
            if !undownloadedModels.contains(where: { $0.filename == filename }) {
                var restoredModel = model
                restoredModel.status = "download"
                restoredModel.rec = canRunModel(model.name)
                undownloadedModels.append(restoredModel)
            }
        }
    }

    private func startAssistantGeneration(
        using inferenceEngine: InferenceEngine,
        messages: [(role: String, content: String)]
    ) async throws {
        await inferenceEngine.setSampling(samplingConfiguration)
        await inferenceEngine.setReasoningEnabled(reasoningEnabled)
        do {
            try await inferenceEngine.generateNext(messages: messages)
        } catch {
            guard case LlamaError.decodeFailed(_) = error else { throw error }
            await inferenceEngine.clear()
            try await inferenceEngine.generateNext(messages: messages)
        }
    }

    private func generateBufferedAssistantTurn(
        using inferenceEngine: InferenceEngine,
        messages: [(role: String, content: String)]
    ) async throws -> String {
        try await startAssistantGeneration(using: inferenceEngine, messages: messages)
        var parts: [String] = []

        while await !inferenceEngine.isComplete && !isStopped {
            if let token = try await inferenceEngine.streamToken() {
                generatedTokenCount += 1
                parts.append(token)
            }
        }

        if !isStopped {
            await inferenceEngine.clearGenerationState()
        }
        return parts.joined()
    }

    private func cleanedAssistantDisplay(_ raw: String) -> String {
        let filter = SpecialTokenFilter()
        let filtered = filter.process(raw) + filter.flush()
        return ReasoningOutput.parse(filtered).answer
    }

    private func toolAwareAssistantResponse(
        using inferenceEngine: InferenceEngine,
        messages initialMessages: [(role: String, content: String)]
    ) async throws -> String {
        var inferenceMessages = initialMessages
        let maximumToolRounds = 3
        var toolRound = 0

        while true {
            let assistantTurn = try await generateBufferedAssistantTurn(
                using: inferenceEngine,
                messages: inferenceMessages
            )
            if isStopped { return "" }

            guard webSearchEnabled,
                  let calls = WebSearchService.parseToolCalls(from: assistantTurn),
                  !calls.isEmpty
            else {
                return assistantTurn
            }

            guard toolRound < maximumToolRounds else {
                throw WebSearchServiceError.invalidResponse
            }

            // Preserve the model's exact canonical JSON call as the assistant
            // message. Training then places one or more raw web result bodies in
            // consecutive "araç" messages.
            inferenceMessages.append((role: "assistant", content: assistantTurn))

            currentResponse = String(localized: "Searching the web…")
            isThinking = false

            for (index, call) in calls.enumerated() {
                if isStopped { return "" }
                let query = call.arguments.query.trimmingCharacters(in: .whitespacesAndNewlines)
                let fallbackID = "call_\(toolRound + 1)_\(index + 1)"
                let callID = call.id?.trimmingCharacters(in: .whitespacesAndNewlines)
                let resolvedCallID = (callID?.isEmpty == false) ? callID! : fallbackID

                let results: [WebSearchResult]
                do {
                    results = try await webSearchService.search(query: query, maxResults: 5)
                } catch {
                    results = []
                }

                let toolBody = WebSearchService.trainingStyleToolResult(
                    query: query,
                    callID: resolvedCallID,
                    results: results
                )
                inferenceMessages.append((role: "tool", content: toolBody))
            }

            toolRound += 1
            currentResponse = ""
            isThinking = true

            // Keep room for another tool call / final answer. Older conversation
            // turns are expendable; the current user request and fresh tool
            // evidence are kept at the end of the prompt.
            let toolBudget = Int(Double(contextSize) * 0.88)
            var tokenCount = await inferenceEngine.countTokens(for: inferenceMessages)
            while tokenCount > toolBudget && inferenceMessages.count > 4 {
                let removeIndex = (inferenceMessages.first?.role == "system") ? 1 : 0
                // Never remove the current assistant tool call or its tool results.
                if removeIndex >= max(0, inferenceMessages.count - (calls.count + 2)) {
                    break
                }
                inferenceMessages.remove(at: removeIndex)
                tokenCount = await inferenceEngine.countTokens(for: inferenceMessages)
            }
        }
    }

    func complete(text: String) async {
        guard !isGenerating else { return }
        if modelSuspendedForSpeech {
            let notice = ChatMessage(
                content: "The language model is reloading after transcription. Please wait a moment and try again.",
                isUser: false,
                timestamp: Date()
            )
            messages.append(notice)
            saveCurrentConversation()
            return
        }
        if inferenceEngine == nil {
            let loaded = await ensureModelLoaded()
            guard loaded, inferenceEngine != nil else {
                let detail = modelLoadError ?? String(localized: "Unknown error")
                let notice = ChatMessage(
                    content: "Could not load the model (\(detail)). Open Manage Models to download one.",
                    isUser: false,
                    timestamp: Date()
                )
                messages.append(notice)
                saveCurrentConversation()
                return
            }
        }

        guard let inferenceEngine else { return }
        await inferenceEngine.setReasoningEnabled(reasoningEnabled)

        speechSynthesizer.stop()
        isGenerating = true
        isStopped = false
        generationStartedAt = Date()
        generatedTokenCount = 0
        rawResponse = ""
        rawResponseParts = []
        displayParts = []
        displayPartsSinceLastFlush = 0
        tokenFilter.reset()
        thinkStripper.reset()
        await inferenceEngine.resume()

        let userMessage = ChatMessage(content: text, isUser: true, timestamp: Date())
        messages.append(userMessage)

        currentResponse = ""
        contextTruncated = false
        isThinking = true

        do {
            // Build full conversation history for multi-turn
            var chatMessages: [(role: String, content: String)] = []
            if webSearchEnabled {
                chatMessages.append((
                    role: "system",
                    content: WebSearchService.augmentedSystemPrompt(base: systemPrompt)
                ))
            } else if !systemPrompt.isEmpty {
                chatMessages.append((role: "system", content: systemPrompt))
            }
            await appendTranscriptContext(to: &chatMessages)
            appendDocumentContext(to: &chatMessages, query: text)
            for msg in messages {
                chatMessages.append((role: msg.isUser ? "user" : "assistant",
                                     content: msg.isUser ? msg.content : msg.displayContent))
            }

            // Context overflow handling
            let budgetFraction = webSearchEnabled ? 0.72 : 0.95
            let budget = Int(Double(contextSize) * budgetFraction)
            var currentTokenCount = await inferenceEngine.countTokens(for: chatMessages)
            lastPromptTokenCount = currentTokenCount
            contextTokenCount = currentTokenCount
            var trimmed = false
            while currentTokenCount > budget && chatMessages.count > 1 {
                // Remove oldest non-system message
                let removeIndex = (chatMessages.first?.role == "system") ? 1 : 0
                if removeIndex >= chatMessages.count { break }
                chatMessages.remove(at: removeIndex)
                trimmed = true
                currentTokenCount = await inferenceEngine.countTokens(for: chatMessages)
            }
            // If still over budget with system + last user, drop system
            if currentTokenCount > budget, chatMessages.count > 1, chatMessages.first?.role == "system" {
                chatMessages.removeFirst()
                currentTokenCount = await inferenceEngine.countTokens(for: chatMessages)
            }
            contextTokenCount = currentTokenCount
            if trimmed {
                contextTruncated = true
                print("Context truncated: \(chatMessages.count) messages remaining")
            }

            // Fix 4: Guard empty chatMessages after trimming
            guard !chatMessages.isEmpty else {
                await MainActor.run {
                    self.currentResponse = "Error: Context too large, no messages fit within budget."
                    let errorMessage = ChatMessage(content: self.currentResponse, isUser: false, timestamp: Date())
                    self.messages.append(errorMessage)
                    self.currentResponse = ""
                    self.isThinking = false
                    self.isGenerating = false
                    self.saveCurrentConversation()
                }
                return
            }

            let savedRaw = try await toolAwareAssistantResponse(
                using: inferenceEngine,
                messages: chatMessages
            )

            // stop() already updates generation/UI state.
            if isStopped { return }

            let finalDisplay = cleanedAssistantDisplay(savedRaw)
            rawResponse = savedRaw
            rawResponseParts = [savedRaw]
            displayParts = finalDisplay.isEmpty ? [] : [finalDisplay]

            var confidence: Float = 1.0
            if let mercanEngine = inferenceEngine as? MercanRuntimeEngine {
                let avg = await mercanEngine.averageEntropy
                confidence = max(0, min(1, 1.0 - (avg / 12.0)))
            }

            await MainActor.run {
                self.currentResponse = finalDisplay
                self.modelConfidence = confidence

                if !savedRaw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    let aiMessage = ChatMessage(content: savedRaw, isUser: false, timestamp: Date())
                    self.messages.append(aiMessage)
                }

                if !finalDisplay.isEmpty {
                    self.speechSynthesizer.feed(
                        displayText: finalDisplay,
                        enabled: self.speakRepliesEnabled
                    )
                }
                self.currentResponse = ""
                self.speechSynthesizer.finish(enabled: self.speakRepliesEnabled)
                self.saveCurrentConversation()
                self.isThinking = false
                self.isGenerating = false
                self.finalizeGenerationStats()
            }

            // Generate title after first exchange (user + assistant = 2 messages)
            if messages.count == 2 {
                await generateTitle(
                    userMessage: text,
                    assistantMessage: savedRaw
                )
            }
        } catch {
            await MainActor.run {
                self.finalizeGenerationStats()
                self.currentResponse = "Error: \(error.localizedDescription)"
                let errorMessage = ChatMessage(content: self.currentResponse, isUser: false, timestamp: Date())
                self.messages.append(errorMessage)
                self.currentResponse = ""
                self.isThinking = false
                self.isGenerating = false
                self.saveCurrentConversation()
            }
        }
    }

    func clear() async {
        saveCurrentConversation()
        speechSynthesizer.stop()
        if let inferenceEngine {
            await inferenceEngine.clear()
        }
        messages = []
        currentResponse = ""
        contextTokenCount = 0
        lastPromptTokenCount = 0
        attachedDocument = nil
        documentImportError = nil
        setTranscriptContext(text: nil, jobId: nil)
        currentConversation = conversationManager?.createNew()
    }

    func stop() async {
        guard let inferenceEngine else {
            return
        }

        isStopped = true
        speechSynthesizer.stop()
        await inferenceEngine.stop()

        // Flush filter/stripper buffers before saving
        let filterFlush = tokenFilter.flush()
        rawResponseParts.append(filterFlush)
        let displayFlush = thinkStripper.process(filterFlush) + thinkStripper.flush()
        if !displayFlush.isEmpty {
            displayParts.append(displayFlush)
        }

        rawResponse = rawResponseParts.joined()
        currentResponse = displayParts.joined()

        // Save raw content for accurate multi-turn history (fall back to display if empty)
        let contentToSave = rawResponse.isEmpty ? currentResponse : rawResponse
        if !contentToSave.isEmpty {
            let aiMessage = ChatMessage(content: contentToSave, isUser: false, timestamp: Date())
            messages.append(aiMessage)
            currentResponse = ""
        }
        isThinking = false
        isGenerating = false
        finalizeGenerationStats()
        saveCurrentConversation()
    }

    private func finalizeGenerationStats() {
        guard let started = generationStartedAt else { return }
        let duration = max(Date().timeIntervalSince(started), 0.001)
        lastGenerationDuration = duration
        lastGenerationTokensPerSecond = Double(generatedTokenCount) / duration
        generationStartedAt = nil
    }

    func getTotalRAMInGiB() -> Double {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        return Double(totalMemory) / (1024 * 1024 * 1024)
    }

    let modelRequirements: [String: Double] = [
        "Mercan 0.8B SFT Q4_K_M (482 MiB)": 1.2
    ]

    func canRunModel(_ modelName: String) -> Bool {
        guard let requiredRAM = modelRequirements[modelName] else {
            return true
        }
        let totalRAM = getTotalRAMInGiB()
        return totalRAM >= requiredRAM
    }

    var contextUsageFraction: Double {
        guard contextSize > 0 else { return 0 }
        return min(1, Double(contextTokenCount) / Double(contextSize))
    }

    var currentGenerationPreset: GenerationPreset {
        let values: [(GenerationPreset, Double)] = GenerationPreset.allCases.map { preset in
            let c = preset.configuration
            let distance =
                abs(Double(c.temperature) - temperature) +
                abs(Double(c.topP) - topP) +
                abs(Double(c.topK) - Double(topK)) / 100.0 +
                abs(Double(c.repeatPenalty) - repeatPenalty)
            return (preset, distance)
        }
        return values.min(by: { $0.1 < $1.1 })?.0 ?? .balanced
    }

    func recommendedContextSize() -> UInt32 {
        let ram = getTotalRAMInGiB()
        if ram < 4 { return 2048 }
        if ram < 6 { return 4096 }
        if ram < 8 { return 8192 }
        return 16384
    }

    func optimizeForDevice() {
        contextSize = recommendedContextSize()
        applyGenerationPreset(.balanced)
    }

    func applyGenerationPreset(_ preset: GenerationPreset) {
        let config = preset.configuration
        temperature = Double(config.temperature)
        topK = Int(config.topK)
        topP = Double(config.topP)
        minP = Double(config.minP)
        repeatPenalty = Double(config.repeatPenalty)
    }

    func attachDocument(_ document: AttachedDocument) {
        attachedDocument = document
        documentImportError = nil
        contextTokenCount = 0
    }

    func clearAttachedDocument() {
        attachedDocument = nil
        contextTokenCount = 0
    }

    func localModelSize(filename: String) -> Int64 {
        fileSize(at: getDocumentsDirectory().appendingPathComponent(filename))
    }

    func regenerateLastResponse() async {
        guard !isGenerating,
              let lastUserIndex = messages.lastIndex(where: { $0.isUser }) else { return }
        let prompt = messages[lastUserIndex].content
        messages = Array(messages.prefix(lastUserIndex))
        saveCurrentConversation()
        await complete(text: prompt)
    }

    func runBenchmark() async {
        guard !isGenerating, !isBenchmarking else { return }
        if inferenceEngine == nil {
            guard await ensureModelLoaded() else { return }
        }
        guard let inferenceEngine else { return }

        isBenchmarking = true
        benchmarkResult = nil
        let restoreMessages = await chatMessagesForInference()
        let benchmarkMessages: [(role: String, content: String)] = [
            (role: "system", content: "Answer concisely and directly."),
            (role: "user", content: "Explain in three short sentences why running an AI model locally on a phone can be useful.")
        ]

        let benchmarkSampling = SamplingConfiguration(
            temperature: 0,
            topK: 1,
            topP: 1,
            minP: 0,
            repeatPenalty: 1,
            repeatLastN: 0
        )

        var count = 0
        let started = Date()

        do {
            await inferenceEngine.setSampling(benchmarkSampling)
            await inferenceEngine.resume()
            try await inferenceEngine.generateNext(messages: benchmarkMessages)

            while await !inferenceEngine.isComplete && count < 64 {
                if try await inferenceEngine.streamToken() != nil {
                    count += 1
                }
            }
            await inferenceEngine.stop()

            let duration = max(Date().timeIntervalSince(started), 0.001)
            benchmarkResult = BenchmarkResult(
                modelName: currentModelName.isEmpty ? "Mercan" : currentModelName,
                generatedTokens: count,
                duration: duration,
                tokensPerSecond: Double(count) / duration,
                date: Date()
            )

            await inferenceEngine.setSampling(samplingConfiguration)
            if !restoreMessages.isEmpty {
                try await inferenceEngine.encodePrompt(messages: restoreMessages)
            } else {
                await inferenceEngine.clear()
            }
        } catch {
            modelLoadError = "Benchmark failed: \(error.localizedDescription)"
            await inferenceEngine.setSampling(samplingConfiguration)
        }

        isBenchmarking = false
    }

    // MARK: - Title Generation

    private func generateTitle(userMessage: String, assistantMessage: String) async {
        guard let inferenceEngine else { return }

        let titleMessages: [(role: String, content: String)] = [
            (role: "system", content: "Generate a 3-5 word title for this conversation. Reply with ONLY the title, nothing else."),
            (role: "user", content: userMessage),
            (role: "assistant", content: String(assistantMessage.prefix(200))),
            (role: "user", content: "Generate a short title for this conversation.")
        ]

        let conversationMessages = await chatMessagesForInference()

        do {
            await inferenceEngine.clearGenerationState()
            await inferenceEngine.resume()
            await inferenceEngine.setSampling(samplingConfiguration)
            try await inferenceEngine.generateNext(messages: titleMessages)

            let titleFilter = SpecialTokenFilter()
            let titleThinkStripper = ThinkTagStripper()
            var titleResult = ""

            while await !inferenceEngine.isComplete {
                if let token = try await inferenceEngine.streamToken() {
                    let filtered = titleFilter.process(token)
                    if !filtered.isEmpty {
                        let display = titleThinkStripper.process(filtered)
                        titleResult += display
                    }
                    // Cap at 60 chars
                    if titleResult.count >= 60 { break }
                }
            }

            // Fix 3: Stop engine if we broke out early, before clearing
            await inferenceEngine.stop()

            // Flush
            let flush = titleFilter.flush()
            titleResult += titleThinkStripper.process(flush) + titleThinkStripper.flush()

            // Restore KV cache for the user's conversation
            if !conversationMessages.isEmpty {
                try await inferenceEngine.encodePrompt(messages: conversationMessages)
            } else {
                await inferenceEngine.clearGenerationState()
            }

            // Clean up the title
            var title = titleResult
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\"", with: "")
                .replacingOccurrences(of: "\n", with: " ")

            if title.count > 60 {
                title = String(title.prefix(60))
            }

            // Fallback if empty or garbage
            if title.isEmpty || title.count < 2 {
                title = String(userMessage.prefix(40))
            }

            await MainActor.run {
                if var conversation = self.currentConversation {
                    conversation.title = title
                    self.currentConversation = conversation
                    self.conversationManager?.save(conversation)
                }
            }
        } catch {
            // Fallback: use truncated user message
            let fallback = String(userMessage.prefix(40))
            await MainActor.run {
                if var conversation = self.currentConversation {
                    conversation.title = fallback
                    self.currentConversation = conversation
                    self.conversationManager?.save(conversation)
                }
            }
        }
    }

    // MARK: - Conversation Management

    func saveCurrentConversation() {
        guard !messages.isEmpty else { return }

        if var conversation = currentConversation {
            conversation.messages = messages
            if let conversationTranscript {
                if let relative = try? saveTranscriptToDisk(conversationId: conversation.id, transcript: conversationTranscript) {
                    conversation.transcriptFilename = relative
                    conversation.transcript = nil
                } else {
                    conversation.transcript = conversationTranscript
                }
            }
            conversation.updatedAt = Date()
            conversationManager?.save(conversation)
            currentConversation = conversation
        } else {
            var newConversation = Conversation(messages: messages)
            newConversation.updatedAt = Date()
            conversationManager?.save(newConversation)
            currentConversation = newConversation
            conversationManager?.currentConversationId = newConversation.id
        }
    }

    func loadConversation(id: UUID) {
        saveCurrentConversation()
        attachedDocument = nil
        documentImportError = nil
        contextTokenCount = 0
        guard let conversation = conversationManager?.loadFullConversation(id: id) else { return }
        messages = conversation.messages
        currentConversation = conversation
        conversationManager?.currentConversationId = conversation.id

        if let inline = conversation.transcript, !inline.isEmpty {
            setTranscriptContext(text: inline, jobId: nil)
        } else if conversation.transcriptFilename != nil {
            setTranscriptContext(text: nil, jobId: nil)
            let conversationId = conversation.id
            Task {
                _ = await loadResolvedTranscriptText()
                await MainActor.run {
                    guard self.currentConversation?.id == conversationId else { return }
                    self.refreshAttachedVideoThumbnail()
                }
            }
        } else {
            setTranscriptContext(text: nil, jobId: nil)
        }
        refreshAttachedVideoThumbnail()
    }

    func startNewConversation() async {
        await clear()
    }
}
