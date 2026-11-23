# Mercan iOS

The iOS application source is checked into this repository.

- `Mercan/`: SwiftUI application source and resources.
- `Mercan.xcodeproj/`: Xcode project.
- `scripts/prepare-frameworks.sh`: builds the libmercan iOS runtime and prepares the pinned whisper framework.
- `scripts/build-unsigned.sh`: signing-independent device build.
- `scripts/archive-appstore.sh`: signed App Store Connect archive/export.

The UI source is based on Silo commit `1be4fa9aad0ec72dbb6f40ac07d537ea16648944`; its MIT license is preserved in `THIRD_PARTY_SILO_LICENSE`.

App changes should now be made directly under `ios-app/Mercan/`. Language-model execution flows through `MercanRuntime.xcframework` (libmercan); CI no longer edits Swift source from YAML.

## LFM reasoning

With an LFM3 `.mercan` model loaded, use the brain icon in the composer or the **Reasoning (LFM)** switch in Settings. The preference persists on-device and starts **off**. Generated `<think>...</think>` text appears in an expandable Reasoning panel; the final answer stays separate. Turning reasoning off asks the model to skip thinking using the upstream `<think></think>` prefix, but compliance is model-dependent. Regular chat streams the reasoning panel live. Web-grounded generations buffer candidates until validated.


## EMA Lightning Turkish voice (experimental)

This branch adds a second, independent on-device inference engine for speech:

- **LLM:** the existing LFM3 `.mercan` model continues to produce text in `MercanRuntime.xcframework`.
- **TTS:** EMA Lightning synthesizes complete *answer* sentences from the live LFM stream. Text inside `<think>…</think>` is never sent to speech.
- **Setup:** Settings → Speech → Read answers aloud → Install EMA Lightning → Voice: EMA Lightning.
- **Compatibility:** the available Core ML conversion targets **iOS 26+**. Earlier iPhones can still use the built-in system speech synthesizer. The app's deployment target remains iOS 18.2.
- **Storage:** ~32.6 MB is downloaded once, SHA-256 verified, compiled with Core ML and cached under Application Support. The existing LFM model remains loaded.
- **Fallback:** if the EMA model is missing or its runtime contract disagrees with the published conversion, an error is shown in Speech settings and system speech reads the text instead.

The on-device EMA Core ML runner is a new integration and **needs physical-device testing** against the publisher's model. The app currently provides a lightweight Turkish number expansion, not full `normalizer-tr` parity for every date, currency or abbreviation. The original EMA Lightning is **5.6M acoustic + 3M vocoder parameters**, and emits 48 kHz mono synthetic audio.

Model/code by Canberk Aslan, [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning), Apache-2.0. Core ML distribution by Bosphorus Intelligence LLC, [elphi-si/elphi](https://huggingface.co/elphi-si/elphi), Apache-2.0. The model is downloaded from the publisher on the user's request rather than bundled in the application. Include license/NOTICE terms when redistributing these models.

## TestFlight variables

Set these GitHub repository variables when the Apple account is ready:

- `ENABLE_TESTFLIGHT=true`
- `IOS_BUNDLE_ID`
- `APPLE_TEAM_ID`
- `APPSTORE_ISSUER_ID`
- `APPSTORE_API_KEY_ID`

Set these repository secrets:

- `APPSTORE_API_PRIVATE_KEY`
- `APPSTORE_CERTIFICATES_FILE_BASE64`
- `APPSTORE_CERTIFICATES_PASSWORD`

The App Store provisioning profile is expected to be named `AppStore <bundle-id>`.
