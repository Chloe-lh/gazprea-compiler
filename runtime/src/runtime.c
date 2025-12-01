#include <stdio.h>
#include <stdbool.h>
#include "../include/run_time_errors.h"

// Add function named dummyPrint with signature void(int) to llvm to have this linked in.
void dummyPrint(int i) {
  printf("I'm a function! %d\n", i);
}

// Functions for reading from std_input
void readInt(int* i) {
  scanf("%d", i);
}

void readReal(float* f) {
  scanf("%f", f);
}

void readChar(char* c) {
  scanf(" %c", c);
}

void readBool(bool* b) {
  scanf("%d", b);
}

// Returns 0 if stream is good, 1 if Error, 2 if EOF.
// Takes a dummy pointer because the language signature is 'var input_stream' (pass-by-ref)
int32_t stream_state_runtime(int32_t* dummy_stream_handle) {
  if (ferror(stdin)) {
      return 1;
  }
  if (feof(stdin)) {
    return 2;
  }
  return 0;
}
