#!/bin/zsh

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input.gaz or .in file>" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="$1"

if [ ! -f "$INPUT" ]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

# SET YOUR PATHS HERE
# -------------------------------
GAZC="$ROOT_DIR/bin/gazc"
LLC="$ROOT_DIR/../packages/llvm-project/build/bin/llc"
CLANG="/usr/bin/clang"
# -------------------------------

if [ ! -x "$GAZC" ]; then
  echo "gazc not found at $GAZC – build the project first." >&2
  exit 1
fi
if [ ! -x "$LLC" ]; then
  echo "llc not found at $LLC – update the path in run_test.sh." >&2
  exit 1
fi

BASENAME="$(basename "$INPUT")"
STEM="${BASENAME%.*}"

LLVM_IR="$ROOT_DIR/gaz.ll"
OBJ="$ROOT_DIR/gaz.o"
EXE="$ROOT_DIR/gaz"

echo "=== gazc ==="
"$GAZC" --verbose-errors "$INPUT" "$LLVM_IR"

echo "=== llc ==="
"$LLC" -filetype=obj "$LLVM_IR" -o "$OBJ"

echo "=== clang ==="
"$CLANG" "$OBJ" -o "$EXE" -L"$ROOT_DIR/bin" -lgazrt

echo "=== run ($STEM) ==="
DYLD_LIBRARY_PATH="$ROOT_DIR/bin:${DYLD_LIBRARY_PATH:-}" "$EXE"

