#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include "../include/run_time_errors.h"

// Add function named dummyPrint with signature void(int) to llvm to have this linked in.
void dummyPrint(int i) {
  printf("I'm a function! %d\n", i);
}

// Functions for reading from std_input
// as well as checking the stream state

// Global State: 0 = Success, 1 = Error, 2 = EOF
int32_t g_stream_state = 0;

// Helper to check for EOF/Error without changing stream state
void check_stream() {
  int c = fgetc(stdin);
  if (c == EOF) {
      g_stream_state = 2; // EOF
  } else {
      ungetc(c, stdin);
  }
}

void readInt(int32_t* out) {
    int result = scanf("%d", out);
    if (result == 1) {
        g_stream_state = 0;
    } else if (result == 0) {
        g_stream_state = 1; // error
    } else {
        g_stream_state = 2; // EOF
    }
}

void readReal(float* out) {
  int result = scanf("%f", out);
  if (result == 1) {
      g_stream_state = 0;
  } else if (result == 0) {
      g_stream_state = 1;
  } else {
      g_stream_state = 2;
  }
}

void readChar(int8_t* out) {
  int result = scanf("%c", (char*)out);
  if (result == 1) {
      g_stream_state = 0;
  } else {
      g_stream_state = 2;
  }
}

void readBool(int8_t* out) {
  // 1. Manually skip whitespace to find the start of the token
  int c;
  while ((c = fgetc(stdin)) != EOF) {
      if (!isspace(c)) {
          ungetc(c, stdin); // Push back the first non-whitespace
          break;
      }
  }
  
  if (c == EOF) {
      g_stream_state = 2;
      return;
  }

  // 2. Peek at the start character
  c = fgetc(stdin); // This is the non-whitespace char we just found
  ungetc(c, stdin); // Put it back immediately

  // 3. Selective Read
  // Only attempt to consume input if it looks like "true" or "false"
  if (c == 't' || c == 'T' || c == 'f' || c == 'F') {
      char buffer[16];
      // Read string to match. This consumes the token.
      // If it looked like a bool but wasn't (e.g. "tree"), we consume it and error.
      if (scanf("%15s", buffer) == 1) {
           if (strcmp(buffer, "true") == 0) {
              *out = 1;
              g_stream_state = 0;
          } else if (strcmp(buffer, "false") == 0) {
              *out = 0;
              g_stream_state = 0;
          } else {
              g_stream_state = 1; // Parse error (consumed "tree")
          }
      } else {
          g_stream_state = 2;
      }
  } else {
      // It starts with something else (e.g. '9').
      // Do NOT consume it. Just report error.
      g_stream_state = 1;
  }
}


// Returns the state of the LAST operation
int32_t stream_state_runtime(int32_t dummy) {
    return g_stream_state;
}

