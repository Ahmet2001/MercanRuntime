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
