#!/usr/bin/env bash
# install_engine.sh [SOURCE]: install the TensorFold inkling2 engine as $BIN (config/inkling.env) on BOTH nodes.
#
#   scripts/install_engine.sh            build engine/ with zig (0.17.0) and CUDA, then install
#   scripts/install_engine.sh SOURCE     install a prebuilt binary instead of building
#
# The build needs zig 0.17.0 (engine/.zig-version) and a CUDA 13 toolkit (nvcc). It produces
# engine/zig-out/native/bin/tensorfold-native, with the nvcc-built kernel fatbins embedded (a build without
# -Dnvcc or -Dfatbins produces a host-only binary that refuses to load the model). Knobs for the build:
# NVCC=/path/to/nvcc (default: nvcc on PATH), SM=121 (default; GB10/Spark), FATBINS=/dir (embed prebuilt
# <name>.fatbin images instead of compiling them). The install copies the binary to $BIN on this node and to the
# worker over ssh, and copies the chat template to the worker. It never replaces a running process and never
# touches a GPU. If ENGINE_MD5 is set in config/inkling.env, the installed binary is checked against it;
# otherwise the md5 is printed so you can pin it.
set -euo pipefail
DEP=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
set -a
# shellcheck disable=SC1090
. "${ENV_FILE:-$DEP/config/inkling.env}"
set +a
src=${1:-}
sum() { md5sum < "$1" | cut -d' ' -f1; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }

if [ -z "$src" ]; then
  # Linux: the engine's kernels are nvcc-built fatbins embedded at compile time; without -Dnvcc (or
  # -Dfatbins=DIR with prebuilt images) the build produces a host-only binary that refuses to load the
  # model (the engine errors BuiltWithoutKernels). SM121 is the DGX Spark / GB10 SASS target.
  # NVCC=... overrides the nvcc on PATH; FATBINS=DIR uses prebuilt <name>.fatbin images instead; SM=... overrides 121.
  nvcc_arg=""
  if [ -n "${FATBINS:-}" ]; then
    [ -d "$FATBINS" ] || { echo "FATBINS=$FATBINS is not a directory" >&2; exit 1; }
    nvcc_arg="-Dfatbins=$FATBINS"
    echo "embedding prebuilt kernel fatbins from $FATBINS"
  else
    nvcc=${NVCC:-$(command -v nvcc || true)}
    [ -n "$nvcc" ] || { echo "nvcc not found: install the CUDA 13 toolkit (or set NVCC=/path/to/nvcc, or FATBINS=DIR with prebuilt fatbins)" >&2; exit 1; }
    nvcc_arg="-Dnvcc=$nvcc -Dsm=${SM:-121}"
    echo "building the kernel fatbins and the engine (nvcc: $nvcc, sm ${SM:-121}) ..."
  fi
  command -v zig >/dev/null || { echo "zig not found: install zig 0.17.0 (engine/.zig-version)" >&2; exit 1; }
  echo "building the engine (zig build native -Doptimize=ReleaseSafe $nvcc_arg) ..."
  (cd "$DEP/engine" && zig build native -Doptimize=ReleaseSafe $nvcc_arg)
  src="$DEP/engine/zig-out/native/bin/tensorfold-native"
fi
[ -f "$src" ] || { echo "no binary at $src" >&2; exit 1; }
md5=$(sum "$src")
echo "engine md5: $md5"
if [ -n "$ENGINE_MD5" ] && [ "$md5" != "$ENGINE_MD5" ]; then
  echo "refusing to install: md5 $md5 != ENGINE_MD5 $ENGINE_MD5 (unset ENGINE_MD5 to install anyway)" >&2
  exit 1
fi

mkdir -p "$(dirname "$BIN")"
cp -p "$src" "$BIN.tmp.$$" && mv "$BIN.tmp.$$" "$BIN" && chmod +x "$BIN"
[ "$(sum "$BIN")" = "$md5" ] || { echo "head copy verify failed" >&2; rm -f "$BIN"; exit 1; }
echo "head: installed $BIN"

wssh "mkdir -p '$(dirname "$BIN")'"
rsync -a "$BIN" "$WORKER_SSH:$BIN.tmp.$$"
wssh "mv '$BIN.tmp.$$' '$BIN' && chmod +x '$BIN'"
[ "$(wssh "md5sum < '$BIN' | cut -d' ' -f1")" = "$md5" ] || { echo "worker copy verify failed" >&2; exit 1; }
echo "worker: installed $BIN"

[ "$(sum "$CHAT_TEMPLATE")" = "$TEMPLATE_MD5" ] || { echo "$CHAT_TEMPLATE is not template C ($TEMPLATE_MD5)" >&2; exit 1; }
wssh "mkdir -p '$(dirname "$CHAT_TEMPLATE")'"
rsync -a "$CHAT_TEMPLATE" "$WORKER_SSH:$CHAT_TEMPLATE"
[ "$(wssh "md5sum < '$CHAT_TEMPLATE' | cut -d' ' -f1")" = "$TEMPLATE_MD5" ] || { echo "worker template verify failed" >&2; exit 1; }
echo "worker: template C at $CHAT_TEMPLATE ($TEMPLATE_MD5)"
echo
echo "done. Pin this build by setting ENGINE_MD5=$md5 in config/inkling.env"
