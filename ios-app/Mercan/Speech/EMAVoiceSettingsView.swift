import SwiftUI

/// Installed independently of the .mercan LLM; both engines can stay loaded.
struct EMAVoiceSettingsView: View {
    @ObservedObject var speech: SpeechSynthesizerService

    var body: some View {
        Picker("Voice", selection: $speech.voiceEngine) {
            Text(MercanVoiceEngine.system.title).tag(MercanVoiceEngine.system)
            if speech.isEMASupported && speech.isEMAInstalled {
                Text(MercanVoiceEngine.emaLightning.title).tag(MercanVoiceEngine.emaLightning)
            }
        }
        if speech.isEMASupported {
            if !speech.isEMAInstalled {
                Button {
                    speech.installEMA()
                } label: {
                    Label(speech.isInstallingEMA ? "Installing EMA Lightning…" : "Install EMA Lightning (~33 MB)", systemImage: "arrow.down.circle")
                }
                .disabled(speech.isInstallingEMA)
                if speech.isInstallingEMA {
                    ProgressView(value: speech.installProgress)
                }
            } else {
                Label("EMA Lightning installed · Offline", systemImage: "checkmark.circle")
                    .foregroundStyle(.secondary)
                Button {
                    speech.testEMA()
                } label: {
                    Label("Test EMA voice", systemImage: "waveform")
                }
            }
        } else {
            Text("EMA Lightning Core ML requires iOS 26 or newer. The system voice remains available.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        if let error = speech.emaError {
            Text(error)
                .font(.footnote)
                .foregroundStyle(.red)
                .textSelection(.enabled)
        }
        Text("EMA Lightning is a synthetic Turkish voice by Canberk Aslan (Apache-2.0). Audio is generated locally. LFM reasoning is never spoken.")
            .font(.footnote)
            .foregroundStyle(.secondary)
    }
}
