import SwiftUI
import UIKit

struct MessageBubble: View {
    let message: ChatMessage
    var isLastAssistant = false
    var onRegenerate: (() -> Void)?

    var body: some View {
        VStack(alignment: message.isUser ? .trailing : .leading, spacing: 7) {
            HStack {
                if message.isUser { Spacer(minLength: 44) }

                if message.isUser {
                    Text(message.displayContent)
                        .textSelection(.enabled)
                        .padding(.horizontal, 15)
                        .padding(.vertical, 11)
                        .background(MercanTheme.coral.opacity(0.11))
                        .clipShape(
                            UnevenRoundedRectangle(
                                topLeadingRadius: 18,
                                bottomLeadingRadius: 18,
                                bottomTrailingRadius: 4,
                                topTrailingRadius: 18
                            )
                        )
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        if !message.reasoningContent.isEmpty {
                            ReasoningDisclosure(reasoning: message.reasoningContent)
                        }
                        if !message.displayContent.isEmpty {
                            MarkdownText(text: message.displayContent, isStreaming: false)
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                if !message.isUser { Spacer(minLength: 8) }
            }

            if !message.isUser {
                HStack(spacing: 14) {
                    Button {
                        UIPasteboard.general.string = message.displayContent
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }

                    if isLastAssistant, let onRegenerate {
                        Button(action: onRegenerate) {
                            Label("Regenerate", systemImage: "arrow.clockwise")
                        }
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .buttonStyle(.plain)
            }
        }
    }
}

/// Streaming intentionally uses plain Text. Re-parsing Markdown every 50 ms is
/// expensive on-device; the final persisted assistant bubble still renders full
/// Markdown after generation completes.
struct StreamingBubble: View {
    let content: String

    var body: some View {
        let parsed = ReasoningOutput.parse(content)
        HStack(alignment: .top, spacing: 8) {
            MercanMark(size: 20)
            VStack(alignment: .leading, spacing: 8) {
                if !parsed.reasoning.isEmpty {
                    ReasoningDisclosure(
                        reasoning: parsed.reasoning,
                        initiallyExpanded: true,
                        inProgress: parsed.isInsideReasoning
                    )
                }
                if !parsed.answer.isEmpty {
                    Text(parsed.answer)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// User-controlled disclosure for the model's generated thinking text.
/// The live stream opens by default; saved responses stay collapsed.
struct ReasoningDisclosure: View {
    let reasoning: String
    let inProgress: Bool
    @State private var isExpanded: Bool

    init(reasoning: String, initiallyExpanded: Bool = false, inProgress: Bool = false) {
        self.reasoning = reasoning
        self.inProgress = inProgress
        _isExpanded = State(initialValue: initiallyExpanded)
    }

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            Text(reasoning)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 6)
        } label: {
            Label(inProgress ? "Reasoning…" : "Reasoning", systemImage: "brain.head.profile")
                .font(.subheadline.weight(.medium))
        }
        .padding(12)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

struct ThinkingIndicator: View {
    @State private var dotCount = 0
    let timer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 8) {
            MercanMark(size: 20)
            Text("Thinking")
                .foregroundStyle(.secondary)
                .font(.subheadline)
            HStack(spacing: 3) {
                ForEach(0..<3) { index in
                    Circle()
                        .fill(MercanTheme.coral)
                        .frame(width: 5, height: 5)
                        .opacity(index <= dotCount ? 1 : 0.25)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onReceive(timer) { _ in
            dotCount = (dotCount + 1) % 3
        }
    }
}
