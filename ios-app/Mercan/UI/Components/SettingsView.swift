import SwiftUI

private struct SystemPromptPreset: Identifiable {
    let id: String
    let prompt: String
}

struct SettingsView: View {
    @ObservedObject var llamaState: LlamaState
    @Environment(\.dismiss) private var dismiss

    @State private var showManageModels = false
    @State private var showSystemPromptEditor = false

    private let promptPresets = [
        SystemPromptPreset(id: "Default", prompt: ""),
        SystemPromptPreset(id: "Concise", prompt: "Answer clearly and concisely. Prefer short, direct responses unless detail is requested."),
        SystemPromptPreset(id: "Coder", prompt: "Act as a careful software engineering assistant. Prefer correct, runnable solutions and explain important tradeoffs."),
        SystemPromptPreset(id: "Writer", prompt: "Act as a thoughtful writing assistant. Preserve the user's intent, improve clarity, and avoid unnecessary filler."),
        SystemPromptPreset(id: "Translator", prompt: "Translate faithfully while preserving tone, meaning, formatting, names, and technical terminology.")
    ]

    var body: some View {
        NavigationStack {
            List {
                Section("Model") {
                    Button { showManageModels = true } label: {
                        HStack {
                            Label("Manage Models", systemImage: "cube.box")
                            Spacer()
                            Text("\(llamaState.downloadedModels.count)")
                                .foregroundStyle(.secondary)
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                    }

                    Picker("Context size", selection: $llamaState.contextSize) {
                        Text("2K").tag(UInt32(2048))
                        Text("4K").tag(UInt32(4096))
                        Text("8K").tag(UInt32(8192))
                        Text("16K").tag(UInt32(16384))
                    }

                    Button {
                        llamaState.optimizeForDevice()
                    } label: {
                        Label("Optimize for this iPhone", systemImage: "wand.and.stars")
                    }
                }

                Section("Generation") {
                    Picker("Preset", selection: Binding(
                        get: { llamaState.currentGenerationPreset },
                        set: { llamaState.applyGenerationPreset($0) }
                    )) {
                        ForEach(GenerationPreset.allCases) { preset in
                            Text(preset.title).tag(preset)
                        }
                    }

                    GenerationSlider(title: "Temperature", value: $llamaState.temperature, range: 0...1.5, step: 0.05)
                    GenerationSlider(title: "Top P", value: $llamaState.topP, range: 0.1...1.0, step: 0.05)
                    GenerationSlider(title: "Min P", value: $llamaState.minP, range: 0...0.2, step: 0.01)

                    Stepper(value: $llamaState.topK, in: 1...100) {
                        HStack {
                            Text("Top K")
                            Spacer()
                            Text("\(llamaState.topK)").foregroundStyle(.secondary)
                        }
                    }

                    GenerationSlider(title: "Repeat penalty", value: $llamaState.repeatPenalty, range: 1.0...1.5, step: 0.05)
                }

                Section("Web") {
                    Label("Per-message Web Search", systemImage: "globe")
                    Text("Tap the globe next to the message field before sending. Mercan searches first, injects the fresh web results as grounding context, and only then lets the local model generate the answer. The model no longer decides whether to call web search itself.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("System prompt") {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            ForEach(promptPresets) { preset in
                                Button(preset.id) {
                                    llamaState.systemPrompt = preset.prompt
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }

                    Button { showSystemPromptEditor = true } label: {
                        HStack {
                            Label("Edit custom prompt", systemImage: "text.alignleft")
                            Spacer()
                            Text("\(llamaState.systemPrompt.count) chars")
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Information") {
                    NavigationLink("Privacy") {
                        PrivacyView()
                    }
                    NavigationLink("Diagnostics") {
                        DiagnosticsView(llamaState: llamaState)
                    }
                    NavigationLink("Benchmark") {
                        BenchmarkView(llamaState: llamaState)
                    }
                }

                Section {
                    Text("Responses are generated by a local AI model and may be inaccurate. Verify important information independently.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .sheet(isPresented: $showManageModels) {
            ManageModelsView(llamaState: llamaState)
        }
        .sheet(isPresented: $showSystemPromptEditor) {
            SystemPromptEditorView(llamaState: llamaState)
        }
    }
}

private struct GenerationSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                Spacer()
                Text(value, format: .number.precision(.fractionLength(2)))
                    .foregroundStyle(.secondary)
            }
            Slider(value: $value, in: range, step: step)
                .tint(MercanTheme.coral)
        }
        .padding(.vertical, 2)
    }
}
