#!/bin/sh
# Vendor the pinned libpdfium.so (linux/amd64) into lib/.
#
# PDFium is embedded and dlopen'd at runtime (pdfium.go, loaded from memfd like
# ONNX Runtime). This reproduces the shipped lib/libpdfium.so byte-for-byte from
# the pinned bblanchon/pdfium-binaries release and checks its sha256.
#
#   sh tools/vendor_pdfium.sh
#
# pdfium's public C API (the FPDF_* symbols we resolve with dlsym) is stable, so
# a bump is drop-in — but rendering feeds detect/rec/order, so re-validate the
# guard and the Omni Parse Bench PDF suite before committing a new build (see
# .agents/skills/quality-eval/SKILL.md). To bump: set RELEASE/SHA below to the
# new release tag and the sha256 of the .so inside it.
set -eu

RELEASE=chromium/8076  # pdfium-binaries release == PDFium 156.0.8076.0
SHA256=fd4de9f91f183476d04b1bc3fd17dacd95aba349c55d2c376512d95e082b441e
URL="https://github.com/bblanchon/pdfium-binaries/releases/download/${RELEASE}/pdfium-linux-x64.tgz"

dst="$(dirname "$0")/../lib/libpdfium.so"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "pdfium ${RELEASE}: downloading ${URL}"
curl -fsSL -o "$tmp/pdfium.tgz" "$URL"
tar xzf "$tmp/pdfium.tgz" -C "$tmp" lib/libpdfium.so
echo "${SHA256}  ${tmp}/lib/libpdfium.so" | sha256sum -c -
mkdir -p "$(dirname "$dst")"
cp "$tmp/lib/libpdfium.so" "$dst"
echo "wrote ${dst} (PDFium 156.0.8076.0)"
