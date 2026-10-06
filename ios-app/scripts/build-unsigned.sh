#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="$ROOT/ios-app"
WORK="$ROOT/build/ios"
DERIVED="$WORK/DerivedData-unsigned"
OUT="$WORK/Mercan-iOS-unsigned.ipa"

test -d "$APP_DIR/llama.xcframework" || { echo "Run ios-app/scripts/prepare-frameworks.sh first" >&2; exit 2; }
test -d "$APP_DIR/whisper.xcframework" || { echo "Run ios-app/scripts/prepare-frameworks.sh first" >&2; exit 2; }

rm -rf "$DERIVED" "$WORK/Payload" "$OUT"
xcodebuild \
  -project "$APP_DIR/Mercan.xcodeproj" \
  -scheme Mercan \
  -configuration Release \
  -sdk iphoneos \
  -destination 'generic/platform=iOS' \
  -derivedDataPath "$DERIVED" \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY="" \
  DEVELOPMENT_TEAM="" \
  build

APP="$(find "$DERIVED/Build/Products/Release-iphoneos" -maxdepth 1 -name '*.app' -print -quit)"
test -n "$APP"
mkdir -p "$WORK/Payload"
cp -R "$APP" "$WORK/Payload/"
(cd "$WORK" && /usr/bin/zip -qry "$(basename "$OUT")" Payload)
shasum -a 256 "$OUT"
