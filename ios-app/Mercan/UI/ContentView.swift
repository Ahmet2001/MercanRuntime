import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var llamaState = LlamaState()
    @StateObject private var conversationManager = ConversationManager()

    @State private var inputText = ""
    @State private var webSearchForNextPrompt = false
    @State private var isSearchingWeb = false
    @State private var drawerOffset: CGFloat = 0

    @State private var showSettings = false
    @State private var showManageModels = false
    @State private var showBenchmark = false
    @State private var showDiagnostics = false
    @State private var showPrivacy = false
    @State private var showDocumentImporter = false
    @State private var showOnboarding = false

    @AppStorage("mercan.didCompleteOnboarding") private var didCompleteOnboarding = false

    @FocusState private var isFocused: Bool

    private let drawerWidth: CGFloat = 310

    private var supportedDocumentTypes: [UTType] {
        var values: [UTType] = [.pdf, .plainText, .json, .commaSeparatedText]
        if let markdown = UTType(filenameExtension: "md") {
            values.append(markdown)
        }
        return values
    }

    private var lastAssistantMessageID: UUID? {
        llamaState.messages.last(where: { !$0.isUser })?.id
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                DrawerView(
                    llamaState: llamaState,
                    conversationManager: conversationManager,
                    onClose: closeDrawer,
                    onNewChat: {
                        Task { await llamaState.startNewConversation() }
                        closeDrawer()
                    },
                    onSelectConversation: { summary in
                        llamaState.loadConversation(id: summary.id)
                        closeDrawer()
                    },
                    onDeleteConversation: { summary in
                        if summary.id == llamaState.currentConversation?.id {
                            Task { await llamaState.startNewConversation() }
                        }
                        conversationManager.delete(summary.id)
                    },
                    onManageModels: {
                        closeDrawer()
                        showManageModels = true
                    },
                    onSettings: {
                        closeDrawer()
                        showSettings = true
                    },
                    onBenchmark: {
                        closeDrawer()
                        showBenchmark = true
                    },
                    onDiagnostics: {
                        closeDrawer()
                        showDiagnostics = true
                    },
                    onPrivacy: {
                        closeDrawer()
                        showPrivacy = true
                    }
                )
                .frame(width: drawerWidth)
                .offset(x: -drawerWidth + drawerOffset)
                .zIndex(drawerOffset > 0 ? 2 : -1)

                VStack(spacing: 0) {
                    HeaderView(
                        currentModel: llamaState.currentModelName,
                        models: llamaState.downloadedModels,
                        isLoadingModel: llamaState.isLoadingModel,
                        isDownloading: llamaState.isDownloadingDefault,
                        isGenerating: llamaState.isGenerating,
                        downloadProgress: llamaState.defaultDownloadProgress,
                        onMenuTap: {
                            withAnimation(.easeOut(duration: 0.22)) {
                                drawerOffset = drawerOffset > 0 ? 0 : drawerWidth
                            }
                        },
                        onModelSelect: { model in
                            let fileURL = llamaState.getDocumentsDirectory().appendingPathComponent(model.filename)
                            Task { try? await llamaState.loadModel(modelUrl: fileURL) }
                        },
                        onNewChat: {
                            Task { await llamaState.startNewConversation() }
                        },
                        onManageModels: { showManageModels = true }
                    )

                    ContextStatusBar(llamaState: llamaState)

                    if let modelError = llamaState.modelLoadError,
                       !llamaState.isLoadingModel {
                        ModelErrorCard(
                            message: modelError,
                            onModels: { showManageModels = true }
                        )
                        .padding(.horizontal, 12)
                        .padding(.top, 8)
                    }

                    if let document = llamaState.attachedDocument {
                        DocumentAttachmentChip(
                            document: document,
                            onRemove: { llamaState.clearAttachedDocument() }
                        )
                        .padding(.horizontal, 12)
                        .padding(.top, 8)
                    }

                    if llamaState.messages.isEmpty && !llamaState.isGenerating {
                        Spacer()
                        EmptyStateView { suggestion in
                            inputText = suggestion
                            isFocused = true
                        }
                        Spacer()
                    } else {
                        ScrollViewReader { proxy in
                            ScrollView {
                                LazyVStack(spacing: 18) {
                                    ForEach(llamaState.messages) { message in
                                        MessageBubble(
                                            message: message,
                                            isLastAssistant: message.id == lastAssistantMessageID,
                                            onRegenerate: {
                                                Task { await llamaState.regenerateLastResponse() }
                                            }
                                        )
                                        .id(message.id)
                                    }

                                    if llamaState.isGenerating && llamaState.isThinking && llamaState.currentResponse.isEmpty {
                                        ThinkingIndicator()
                                            .id("thinking")
                                    }

                                    if llamaState.isGenerating && !llamaState.currentResponse.isEmpty {
                                        StreamingBubble(content: llamaState.currentResponse)
                                            .id("streaming")
                                    }
                                }
                                .padding(.horizontal, 16)
                                .padding(.top, 16)
                                .padding(.bottom, 30)
                            }
                            .scrollDismissesKeyboard(.interactively)
                            .onTapGesture { isFocused = false }
                            .onChange(of: llamaState.messages.count) { _, _ in
                                scrollToBottom(proxy)
                            }
                            .onChange(of: llamaState.currentResponse) { _, _ in
                                if llamaState.isGenerating {
                                    withAnimation(.easeOut(duration: 0.15)) {
                                        proxy.scrollTo("streaming", anchor: .bottom)
                                    }
                                }
                            }
                            .onChange(of: llamaState.isThinking) { _, value in
                                if value {
                                    withAnimation(.easeOut(duration: 0.15)) {
                                        proxy.scrollTo("thinking", anchor: .bottom)
                                    }
                                }
                            }
                            .onChange(of: llamaState.currentConversation?.id) { _, _ in
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
                                    scrollToBottom(proxy)
                                }
                            }
                        }
                    }

                    if isSearchingWeb {
                        HStack(spacing: 8) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Searching the web…")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        .padding(.horizontal, 16)
                        .padding(.bottom, 4)
                    }

                    InputComposer(
                        text: $inputText,
                        webSearchForNextPrompt: $webSearchForNextPrompt,
                        isGenerating: llamaState.isGenerating,
                        inputsDisabled: llamaState.isLoadingModel || isSearchingWeb,
                        onSend: { Task { await submitMessage() } },
                        onStop: { Task { await llamaState.stop() } },
                        onDocumentImport: { showDocumentImporter = true },
                        focusState: $isFocused
                    )
                }
                .frame(width: geometry.size.width, height: geometry.size.height)
                .offset(x: drawerOffset)
                .overlay {
                    Color.black
                        .opacity(Double(drawerOffset / drawerWidth) * 0.25)
                        .allowsHitTesting(drawerOffset > 0)
                        .onTapGesture { closeDrawer() }
                }
                .gesture(drawerGesture(width: geometry.size.width))
            }
        }
        .background(Color(.systemBackground))
        .tint(MercanTheme.coral)
        .onAppear {
            llamaState.conversationManager = conversationManager
            // Legacy model-driven tool calls stay disabled. The user explicitly
            // enables Web Search per prompt in the composer instead.
            llamaState.webSearchEnabled = false
            showOnboarding = !didCompleteOnboarding
            Task {
                if !llamaState.isModelLoaded, !llamaState.downloadedModels.isEmpty {
                    _ = await llamaState.ensureModelLoaded()
                }
            }
        }
        .fullScreenCover(isPresented: $showOnboarding) {
            OnboardingView {
                didCompleteOnboarding = true
                showOnboarding = false
            }
            .interactiveDismissDisabled()
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(llamaState: llamaState)
        }
        .sheet(isPresented: $showManageModels) {
            ManageModelsView(llamaState: llamaState)
        }
        .sheet(isPresented: $showBenchmark) {
            NavigationStack { BenchmarkView(llamaState: llamaState) }
        }
        .sheet(isPresented: $showDiagnostics) {
            NavigationStack { DiagnosticsView(llamaState: llamaState) }
        }
        .sheet(isPresented: $showPrivacy) {
            NavigationStack { PrivacyView() }
        }
        .fileImporter(
            isPresented: $showDocumentImporter,
            allowedContentTypes: supportedDocumentTypes,
            allowsMultipleSelection: false
        ) { result in
            handleDocumentImport(result)
        }
    }

    private func closeDrawer() {
        withAnimation(.easeOut(duration: 0.22)) {
            drawerOffset = 0
        }
    }

    private func drawerGesture(width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 12)
            .onChanged { value in
                let translation = value.translation.width
                if drawerOffset == 0 {
                    guard value.startLocation.x < 36 else { return }
                    drawerOffset = min(max(0, translation), drawerWidth)
                } else {
                    drawerOffset = min(max(0, drawerWidth + translation), drawerWidth)
                }
            }
            .onEnded { value in
                let velocity = value.predictedEndTranslation.width - value.translation.width
                withAnimation(.easeOut(duration: 0.22)) {
                    if velocity > 90 || (drawerOffset > drawerWidth * 0.5 && velocity > -90) {
                        drawerOffset = drawerWidth
                    } else {
                        drawerOffset = 0
                    }
                }
            }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        if llamaState.isGenerating && !llamaState.currentResponse.isEmpty {
            proxy.scrollTo("streaming", anchor: .bottom)
        } else if let last = llamaState.messages.last {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }

    @MainActor
    private func submitMessage() async {
        guard !llamaState.isGenerating, !isSearchingWeb else { return }

        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        if llamaState.downloadedModels.isEmpty {
            showManageModels = true
            return
        }

        if !llamaState.isModelLoaded {
            guard await llamaState.ensureModelLoaded() else {
                showManageModels = true
                return
            }
        }

        let shouldSearchWeb = webSearchForNextPrompt
        webSearchForNextPrompt = false
        inputText = ""

        // Always keep the old model-driven tool-call loop disabled. If search is
        // requested, the app performs it before any generation and injects a
        // synthetic assistant tool-call + `araç` result into the inference-only
        // prompt sequence.
        llamaState.webSearchEnabled = false
        if shouldSearchWeb {
            isSearchingWeb = true
            await llamaState.completeWithForcedWebSearch(text: text) {
                isSearchingWeb = false
            }
            isSearchingWeb = false
        } else {
            await llamaState.complete(text: text)
        }
    }

    private func handleDocumentImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            do {
                let document = try DocumentContextService.load(from: url)
                llamaState.attachDocument(document)
            } catch {
                llamaState.documentImportError = error.localizedDescription
                llamaState.modelLoadError = error.localizedDescription
            }

        case .failure(let error):
            llamaState.documentImportError = error.localizedDescription
            llamaState.modelLoadError = error.localizedDescription
        }
    }
}

private struct DocumentAttachmentChip: View {
    let document: AttachedDocument
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "doc.text.fill")
                .foregroundStyle(MercanTheme.coral)

            VStack(alignment: .leading, spacing: 2) {
                Text(document.name)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text("\(document.kind) • \(document.characterCount.formatted()) characters")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .background(MercanTheme.coral.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

private struct ModelErrorCard: View {
    let message: String
    let onModels: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 5) {
                Text("Model needs attention")
                    .font(.subheadline.weight(.semibold))
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Open Models", action: onModels)
                    .font(.caption.weight(.semibold))
            }
            Spacer()
        }
        .padding(12)
        .background(Color.orange.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

#Preview {
    ContentView()
}
