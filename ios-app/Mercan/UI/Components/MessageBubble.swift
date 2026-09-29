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
                    MarkdownText(text: message.displayContent, isStreaming: false)
                        .textSelection(.enabled)
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
        HStack(alignment: .top, spacing: 8) {
            MercanMark(size: 20)
            Text(content.isEmpty ? " " : content)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        }
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
