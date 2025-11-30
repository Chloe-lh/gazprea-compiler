#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: ./input_test.sh <filename_without_extension>"
    echo "Example: ./input_test.sh input_simple"
    exit 1
fi

INPUT_FILE=$1
CWD=$(pwd)

# Stop execution if any command fails
set -e

echo "--- Step 1: Generating LLVM IR (.ll) ---"
bin/gazc "tests/testfiles/streams/${INPUT_FILE}.in" "tests/fake-output/${INPUT_FILE}.ll"

echo "--- Step 2: Compiling to Object File (.o) ---"
llc -filetype=obj "tests/fake-output/${INPUT_FILE}.ll" -o "tests/fake-output/${INPUT_FILE}.o"

echo "--- Step 3: Linking and creating executable ---"
clang "tests/fake-output/${INPUT_FILE}.o" -o "bin/${INPUT_FILE}" -Lbin -lgazrt -Wl,-rpath,"${CWD}/bin"

echo "--- Done. Executable created at bin/${INPUT_FILE} ---"