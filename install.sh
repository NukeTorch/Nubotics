#!/bin/sh
# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
#
# Nubotics installer: NukeTorch 1.0.5, the Nubotics client, the Nubotics model and the text model NukeTorch 1.0.5 serves beside it.
# Everything goes into ~/NukeTorch (or $NUKETORCH_HOME).
#   sh install.sh                 everything
#   sh install.sh --client-only   the client alone (any Mac, Python 3.9 or later)
set -eu

D="${NUKETORCH_HOME:-$HOME/NukeTorch}"
ENGINE_URL="https://github.com/NukeTorch/nuketorch/releases/download/1.0.5/install.sh"
ENGINE="$D/nuketorch-1.0.5-macos-arm64/nuketorch-serve"
SOURCE="${NUBOTICS_SOURCE:-git+https://github.com/NukeTorch/Nubotics}"
PKG="nubotics[gr00t] @ $SOURCE"
[ -d "$SOURCE" ] && PKG="$(cd "$SOURCE" && pwd)[gr00t]"
ENV="$D/nubotics-env"
CLIENT_ONLY=0
[ "${1:-}" = "--client-only" ] && CLIENT_ONLY=1

say() { printf '%s\n' "$*"; }
die() { printf 'nubotics install: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "needs a Mac with Apple silicon"
if [ "$CLIENT_ONLY" = 0 ]; then
  chip="$(sysctl -n machdep.cpu.brand_string)"
  case "$chip" in *"M5 Max"*) ;; *) die "Nubotics needs an M5 Max (this Mac: $chip). For the client alone: sh install.sh --client-only" ;; esac
  v="$(sw_vers -productVersion)"; major="${v%%.*}"; rest="${v#*.}"; minor="${rest%%.*}"
  [ "$major" -gt 27 ] || { [ "$major" -eq 27 ] && [ "${minor:-0}" -ge 2 ]; } || die "needs macOS 27.2 or later (this Mac: $v)"
fi
command -v python3 >/dev/null || die "needs python3 (xcode-select --install)"
mkdir -p "$D"

# 1. NukeTorch: its own installer shows its license and asks you to accept it.
if [ "$CLIENT_ONLY" = 0 ] && [ ! -x "$ENGINE" ]; then
  say "== NukeTorch 1.0.5"
  curl -fsSL "$ENGINE_URL" | sh
  [ -x "$ENGINE" ] || die "NukeTorch was not installed"
fi

# 2. The client.
say "== Nubotics client"
[ -x "$ENV/bin/python" ] || python3 -m venv "$ENV"
"$ENV/bin/pip" install -q -U pip
"$ENV/bin/pip" install -q -U huggingface_hub "$PKG"

# 3. The models.
if [ "$CLIENT_ONLY" = 0 ]; then
  say "== Nubotics model (about 14 GB)"
  "$ENV/bin/hf" download NukeTorch/Nubotics --local-dir "$D/models/Nubotics"
  (cd "$D/models/Nubotics" && shasum -a 256 -c -q SHA256SUMS) || die "the Nubotics model failed its SHA-256 check"
  say "== Text model NukeTorch 1.0.5 serves beside the policy (about 21 GB)"
  "$ENV/bin/hf" download NukeTorch/Qwen3.6-35B-A3B-4bit-DFlash2 --local-dir "$D/models/Qwen3.6-35B-A3B-4bit-DFlash2"
fi

say ""
say "Installed in $D."
if [ "$CLIENT_ONLY" = 0 ]; then
  say "Start the server:            $ENV/bin/nubotics serve"
  say "Then, in another terminal:   $ENV/bin/nubotics demo"
fi
