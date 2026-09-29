import Foundation
import Combine

/// Lightweight UI-facing stream state. The inference runtime can produce one
/// token at a time, while this object is updated only with coalesced text.
@MainActor
final class LiveTokenStreamModel: ObservableObject {
    static let shared = LiveTokenStreamModel()

    @Published private(set) var text = ""

    private init() {}

    fileprivate func replace(with value: String) {
        text = value
    }

    fileprivate func clear() {
        text = ""
    }
}

/// Coalesces raw runtime tokens before they reach SwiftUI.
///
/// Goals:
/// - keep model decoding fully token-by-token and unthrottled;
/// - update SwiftUI at most about every 50 ms (or on a useful text boundary);
/// - hide canonical tool-call JSON until it is classified, so internal tool
///   plumbing never flashes in the chat UI;
/// - strip special/control and hidden <think> content before presentation.
actor LiveTokenCoalescer {
    static let shared = LiveTokenCoalescer()

    private let flushIntervalNanoseconds: UInt64 = 50_000_000
    private let immediateFlushCharacterThreshold = 40
    private let safetyProbeCharacters = 96

    private var generationID = 0
    private var probe = ""
    private var visibleText = ""
    private var charactersSinceFlush = 0
    private var isVisibleAnswer = false
    private var isHiddenToolCall = false
    private var flushTask: Task<Void, Never>?

    private var tokenFilter = SpecialTokenFilter()
    private var thinkStripper = ThinkTagStripper()

    func beginGeneration() async {
        generationID += 1
        flushTask?.cancel()
        flushTask = nil
        probe = ""
        visibleText = ""
        charactersSinceFlush = 0
        isVisibleAnswer = false
        isHiddenToolCall = false
        tokenFilter.reset()
        thinkStripper.reset()
        await MainActor.run {
            LiveTokenStreamModel.shared.clear()
        }
    }

    func ingest(_ token: String) async {
        guard !token.isEmpty else { return }

        if isHiddenToolCall {
            // Keep consuming silently. The caller still needs the exact raw JSON
            // for tool parsing, but the UI must never see it.
            probe += token
            return
        }

        if !isVisibleAnswer {
            probe += token
            let trimmed = probe.trimmingCharacters(in: .whitespacesAndNewlines)
            let canonicalPrefix = #"{"tool_calls":"#

            if trimmed.hasPrefix(canonicalPrefix) {
                isHiddenToolCall = true
                flushTask?.cancel()
                flushTask = nil
                await MainActor.run {
                    LiveTokenStreamModel.shared.clear()
                }
                return
            }

            let beginsLikeJSON = trimmed.first == "{"
            let stillCanonicalPrefix = canonicalPrefix.hasPrefix(trimmed)
            let minimumPlainTextProbe = 24

            // Ordinary prose can start streaming quickly. JSON-looking output is
            // held longer so a synthetic/repeated web_search call cannot flash.
            if stillCanonicalPrefix ||
                (beginsLikeJSON && probe.count < safetyProbeCharacters) ||
                (!beginsLikeJSON && probe.count < minimumPlainTextProbe) {
                return
            }

            isVisibleAnswer = true
            let initial = probe
            probe = ""
            await appendVisibleRaw(initial)
            return
        }

        await appendVisibleRaw(token)
    }

    func finishGeneration() async {
        flushTask?.cancel()
        flushTask = nil

        if isHiddenToolCall {
            await MainActor.run {
                LiveTokenStreamModel.shared.clear()
            }
            return
        }

        if !isVisibleAnswer, !probe.isEmpty {
            let trimmed = probe.trimmingCharacters(in: .whitespacesAndNewlines)
            if WebSearchService.parseToolCalls(from: trimmed) != nil {
                probe = ""
                await MainActor.run {
                    LiveTokenStreamModel.shared.clear()
                }
                return
            }

            isVisibleAnswer = true
            let tail = probe
            probe = ""
            await appendVisibleRaw(tail, scheduleOnly: true)
        }

        let filterTail = tokenFilter.flush()
        let strippedTail = thinkStripper.process(filterTail) + thinkStripper.flush()
        if !strippedTail.isEmpty {
            visibleText += strippedTail
        }
        await flushNow()
    }

    func clearPresentation() async {
        generationID += 1
        flushTask?.cancel()
        flushTask = nil
        probe = ""
        visibleText = ""
        charactersSinceFlush = 0
        isVisibleAnswer = false
        isHiddenToolCall = false
        tokenFilter.reset()
        thinkStripper.reset()
        await MainActor.run {
            LiveTokenStreamModel.shared.clear()
        }
    }

    private func appendVisibleRaw(_ raw: String, scheduleOnly: Bool = false) async {
        let filtered = tokenFilter.process(raw)
        guard !filtered.isEmpty else { return }

        let display = thinkStripper.process(filtered)
        guard !display.isEmpty else { return }

        visibleText += display
        charactersSinceFlush += display.count

        if !scheduleOnly,
           charactersSinceFlush >= immediateFlushCharacterThreshold ||
            display.contains("\n") {
            flushTask?.cancel()
            flushTask = nil
            await flushNow()
            return
        }

        scheduleFlushIfNeeded()
    }

    private func scheduleFlushIfNeeded() {
        guard flushTask == nil else { return }
        let id = generationID
        flushTask = Task {
            try? await Task.sleep(nanoseconds: flushIntervalNanoseconds)
            guard !Task.isCancelled else { return }
            await self.flushIfCurrent(id: id)
        }
    }

    private func flushIfCurrent(id: Int) async {
        guard id == generationID else { return }
        flushTask = nil
        await flushNow()
    }

    private func flushNow() async {
        let snapshot = visibleText
        charactersSinceFlush = 0
        await MainActor.run {
            LiveTokenStreamModel.shared.replace(with: snapshot)
        }
    }
}
