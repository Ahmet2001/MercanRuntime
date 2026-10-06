#!/usr/bin/env bash
set -euo pipefail
: "${IOS_BUNDLE_ID:?IOS_BUNDLE_ID is required}"
: "${APPLE_TEAM_ID:?APPLE_TEAM_ID is required}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="$ROOT/ios-app"
WORK="$ROOT/build/ios"
ARCHIVE="$WORK/Mercan.xcarchive"
EXPORT_DIR="$WORK/export"
EXPORT_PLIST="$WORK/ExportOptions.plist"
PROFILE_NAME="${IOS_PROFILE_NAME:-AppStore $IOS_BUNDLE_ID}"
BUILD_NUMBER="${IOS_BUILD_NUMBER:-1}"
MARKETING_VERSION="${IOS_MARKETING_VERSION:-0.1.0}"

rm -rf "$ARCHIVE" "$EXPORT_DIR"
mkdir -p "$WORK"

xcodebuild \
  -project "$APP_DIR/Mercan.xcodeproj" \
  -scheme Mercan \
  -configuration Release \
  -sdk iphoneos \
  -destination 'generic/platform=iOS' \
  -archivePath "$ARCHIVE" \
  DEVELOPMENT_TEAM="$APPLE_TEAM_ID" \
  PRODUCT_BUNDLE_IDENTIFIER="$IOS_BUNDLE_ID" \
  CODE_SIGN_STYLE=Manual \
  CODE_SIGN_IDENTITY="Apple Distribution" \
  PROVISIONING_PROFILE_SPECIFIER="$PROFILE_NAME" \
  CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
  MARKETING_VERSION="$MARKETING_VERSION" \
  archive

cat > "$EXPORT_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>method</key>
  <string>app-store-connect</string>
  <key>provisioningProfiles</key>
  <dict>
    <key>$IOS_BUNDLE_ID</key>
    <string>$PROFILE_NAME</string>
  </dict>
  <key>signingStyle</key>
  <string>manual</string>
  <key>teamID</key>
  <string>$APPLE_TEAM_ID</string>
</dict>
</plist>
EOF

xcodebuild -exportArchive \
  -archivePath "$ARCHIVE" \
  -exportPath "$EXPORT_DIR" \
  -exportOptionsPlist "$EXPORT_PLIST"

IPA="$(find "$EXPORT_DIR" -maxdepth 1 -name '*.ipa' -print -quit)"
test -n "$IPA"
cp "$IPA" "$WORK/Mercan-AppStore.ipa"
shasum -a 256 "$WORK/Mercan-AppStore.ipa"
