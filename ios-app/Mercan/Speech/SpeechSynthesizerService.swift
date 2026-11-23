import AVFoundation
import Foundation

enum MercanVoiceEngine: String, CaseIterable, Identifiable {
    case system
    case emaLightning

    var id: String { rawValue }

    var title: String {
        switch self {
        case .system: return "System voice"
        case .emaLightning: return "EMA Lightning · Turkish"
        }
    }
}

@MainActor
final class SpeechSynthesizerService: NSObject, ObservableObject {
    @Published private(set) var isSpeaking = false
    @Published private(set) var isInstallingEMA = false
    @Published private(set) var installProgress: Double = 0
    @Published var emaError: String?
    @Published var voiceEngine: MercanVoiceEngine = .system {
        didSet { UserDefaults.standard.set(voiceEngine.rawValue, forKey: "mercanVoiceEngine") }
    }

    var isEMAInstalled: Bool { EMAAssets.isInstalled }
    var isEMASupported: Bool {
        if #available(iOS 26.0, *) { return true }
        return false
    }

    private let synthesizer = AVSpeechSynthesizer()
    private let ema = EMALightningCoreML()
    private let audioEngine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var playerAttached = false
    private var scheduledBuffers = 0
    private var queue: Task<Void, Never>?
    private var generationID = 0
    private var sentenceBuffer = ""
    private var spokenDisplayLength = 0

    override init() {
        super.init()
        synthesizer.delegate = self
        if let stored = UserDefaults.standard.string(forKey: "mercanVoiceEngine"),
           let selected = MercanVoiceEngine(rawValue: stored) {
            voiceEngine = selected
        }
    }

    func installEMA() {
        guard !isInstallingEMA else { return }
        guard isEMASupported else {
            emaError = EMAError.unsupportedSystem.localizedDescription
            return
        }
        isInstallingEMA = true
        emaError = nil
        installProgress = 0
        Task {
            do {
                try await EMAAssets.install { [weak self] progress in
                    Task { @MainActor in self?.installProgress = progress }
                }
                installProgress = 1
                voiceEngine = .emaLightning
            } catch {
                emaError = error.localizedDescription
            }
            isInstallingEMA = false
            objectWillChange.send()
        }
    }

    func testEMA() {
        guard isEMASupported && isEMAInstalled else {
            emaError = EMAError.missingModels.localizedDescription
            return
        }
        stop()
        voiceEngine = .emaLightning
        speak("Merhaba, ben Mercan. Yerel ses modeli hazır.")
    }

    func stop() {
        generationID &+= 1
        queue?.cancel()
        queue = nil
        synthesizer.stopSpeaking(at: .immediate)
        player.stop()
        scheduledBuffers = 0
        sentenceBuffer = ""
        spokenDisplayLength = 0
        isSpeaking = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    /// Called with the final-answer text (never with raw LFM reasoning).
    /// Full sentences are scheduled while the local LLM continues generating.
    func feed(displayText: String, enabled: Bool) {
        guard enabled else { return }
        guard displayText.count > spokenDisplayLength else { return }
        let delta = String(displayText.dropFirst(spokenDisplayLength))
        spokenDisplayLength = displayText.count
        sentenceBuffer += delta
        flushSentences()
    }

    func finish(enabled: Bool) {
        guard enabled else {
            stop()
            return
        }
        let remaining = sentenceBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        sentenceBuffer = ""
        if !remaining.isEmpty { speak(remaining) }
    }

    private func flushSentences() {
        while let range = sentenceBuffer.range(of: #"[\.!\?][\s\n]+"#, options: .regularExpression) {
            let sentence = String(sentenceBuffer[..<range.upperBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            sentenceBuffer.removeSubrange(sentenceBuffer.startIndex..<range.upperBound)
            if sentence.count > 2 { speak(sentence) }
        }
    }

    private func speak(_ text: String) {
        guard !text.isEmpty else { return }
        if voiceEngine == .emaLightning {
            guard isEMASupported, isEMAInstalled else {
                emaError = "Install EMA Lightning from Settings to use this voice."
                speakSystem(text)
                return
            }
            let serial = generationID
            let previous = queue
            queue = Task { [weak self] in
                await previous?.value
                guard let self, !Task.isCancelled, serial == self.generationID else { return }
                do {
                    let samples = try await self.ema.synthesize(text)
                    guard !Task.isCancelled, serial == self.generationID else { return }
                    try self.playEMA(samples)
                } catch {
                    guard serial == self.generationID else { return }
                    self.emaError = error.localizedDescription
                    // No silent loss of the live response on a model mismatch.
                    self.speakSystem(text)
                }
            }
        } else {
            speakSystem(text)
        }
    }

    private func playEMA(_ samples: [Float]) throws {
        guard !samples.isEmpty else { return }
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try session.setActive(true)

        let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!
        if !playerAttached {
            audioEngine.attach(player)
            audioEngine.connect(player, to: audioEngine.mainMixerNode, format: format)
            playerAttached = true
        }
        if !audioEngine.isRunning { try audioEngine.start() }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else {
            throw EMAError.incompatibleModel("Could not allocate the audio buffer.")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { ptr in
            if let base = ptr.baseAddress { channel.update(from: base, count: samples.count) }
        }
        scheduledBuffers += 1
        isSpeaking = true
        player.scheduleBuffer(buffer) { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.scheduledBuffers = max(0, self.scheduledBuffers - 1)
                if self.scheduledBuffers == 0 { self.isSpeaking = false }
            }
        }
        player.play()
    }

    private func speakSystem(_ text: String) {
        guard !text.isEmpty else { return }
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? session.setActive(true)
        let utterance = AVSpeechUtterance(string: text)
        let language = Locale.preferredLanguages.first ?? Locale.current.identifier
        utterance.voice = AVSpeechSynthesisVoice(language: language)
            ?? AVSpeechSynthesisVoice(language: "tr-TR")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        isSpeaking = true
        synthesizer.speak(utterance)
    }
}

extension SpeechSynthesizerService: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            if !self.synthesizer.isSpeaking && self.scheduledBuffers == 0 { self.isSpeaking = false }
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            if self.scheduledBuffers == 0 { self.isSpeaking = false }
        }
    }
}
