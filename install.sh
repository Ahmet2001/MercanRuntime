#!/bin/sh
set -eu

REPO="${MERCAN_GITHUB_REPO:-Ahmet2001/MercanRuntime}"
VERSION="${MERCAN_VERSION:-latest}"
PREFIX="${MERCAN_PREFIX:-/usr/local}"
FORCE_CPU="${MERCAN_FORCE_CPU:-0}"

command -v curl >/dev/null 2>&1 || { echo "Mercan installer requires curl." >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo "Mercan installer requires tar." >&2; exit 1; }
command -v install >/dev/null 2>&1 || { echo "Mercan installer requires install (coreutils)." >&2; exit 1; }

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ;;
  *) echo "Mercan v0.1.x currently supports Linux x86_64 only (detected: $ARCH)." >&2; exit 1 ;;
esac

VARIANT="cpu"
if [ "$FORCE_CPU" != "1" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  VARIANT="cuda"
fi

ASSET="mercan-linux-x86_64-$VARIANT.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

asset_url() {
  asset="$1"
  if [ "$VERSION" = "latest" ]; then
    printf '%s\n' "https://github.com/$REPO/releases/latest/download/$asset"
  else
    printf '%s\n' "https://github.com/$REPO/releases/download/$VERSION/$asset"
  fi
}

echo "Mercan backend: $VARIANT"
echo "Downloading Mercan CLI from $REPO ($VERSION)..."
URL="$(asset_url "$ASSET")"
if ! curl -fL --retry 3 "$URL" -o "$TMP/$ASSET"; then
  if [ "$VARIANT" = "cuda" ]; then
    echo "CUDA package unavailable; falling back to CPU build." >&2
    VARIANT="cpu"
    ASSET="mercan-linux-x86_64-cpu.tar.gz"
    URL="$(asset_url "$ASSET")"
    curl -fL --retry 3 "$URL" -o "$TMP/$ASSET"
  else
    exit 1
  fi
fi

tar -xzf "$TMP/$ASSET" -C "$TMP"
SRC=""
for candidate in "$TMP"/*; do
  if [ -x "$candidate/bin/mercan" ]; then
    SRC="$candidate"
    break
  fi
done

if [ -z "$SRC" ]; then
  echo "Mercan release archive is invalid: bin/mercan not found." >&2
  exit 1
fi

SUDO=""
if [ -d "$PREFIX" ]; then
  if [ ! -w "$PREFIX" ]; then SUDO="sudo"; fi
else
  PARENT="$(dirname "$PREFIX")"
  if [ ! -w "$PARENT" ]; then SUDO="sudo"; fi
fi

if [ -n "$SUDO" ]; then
  command -v sudo >/dev/null 2>&1 || {
    echo "No permission to install into $PREFIX and sudo is unavailable." >&2
    echo "Try a user-writable prefix, for example MERCAN_PREFIX=$HOME/.local" >&2
    exit 1
  }
fi

$SUDO mkdir -p "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/include"
$SUDO install -m 0755 "$SRC/bin/mercan" "$PREFIX/bin/mercan"
if [ -f "$SRC/lib/libmercan.a" ]; then
  $SUDO install -m 0644 "$SRC/lib/libmercan.a" "$PREFIX/lib/libmercan.a"
fi
if [ -f "$SRC/include/mercan.h" ]; then
  $SUDO install -m 0644 "$SRC/include/mercan.h" "$PREFIX/include/mercan.h"
fi
if [ -d "$SRC/share/licenses/mercan" ]; then
  $SUDO mkdir -p "$PREFIX/share/licenses/mercan"
  for license in "$SRC/share/licenses/mercan"/*; do
    $SUDO install -m 0644 "$license" "$PREFIX/share/licenses/mercan/"
  done
fi

echo "Installed Mercan CLI ($VARIANT): $PREFIX/bin/mercan"
if command -v mercan >/dev/null 2>&1; then
  mercan --version
else
  echo "If mercan is not found, add $PREFIX/bin to PATH."
fi
echo "Run: mercan run MercanAI/Mercan-0.8B-SFT"
