#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include "../include/run_time_errors.h"

// Add function named dummyPrint with signature void(int) to llvm to have this linked in.
void dummyPrint(int i) {
  fprintf(stderr, "I'm a function! %d\n", i);
}

// Format functions
char* gazrt_format_int(int32_t val) {
    char* buffer = (char*)malloc(16); // Sufficient for 32-bit int
    if (!buffer) return NULL;
    sprintf(buffer, "%d", val);
    return buffer;
}

char* gazrt_format_real(float val) {
    char* buffer = (char*)malloc(32); // Sufficient for float
    if (!buffer) return NULL;
    sprintf(buffer, "%g", val);
    return buffer;
}

char* gazrt_format_char(int8_t val) {
    char* buffer = (char*)malloc(2);
    if (!buffer) return NULL;
    buffer[0] = (char)val;
    buffer[1] = '\0';
    return buffer;
}

char* gazrt_format_bool(int8_t val) {
    if (val) {
        char* buffer = (char*)malloc(5);
        if (!buffer) return NULL;
        strcpy(buffer, "true");
        return buffer;
    } else {
        char* buffer = (char*)malloc(6);
        if (!buffer) return NULL;
        strcpy(buffer, "false");
        return buffer;
    }
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

// Slice struct for array slicing
typedef struct {
    int32_t* ptr;
    size_t len;
} gaz_slice_int;

// Array slicing function with bounds checking
// All parameters are 0-based (converted from 1-based in MLIR)
// Returns a slice struct with pointer and length
gaz_slice_int gaz_slice_int_checked(int32_t* base, size_t base_len, size_t start, size_t end) {
    gaz_slice_int result;
    
    // If end <= start, return empty slice (no error)
    if (end <= start) {
        result.ptr = base;
        result.len = 0;
        return result;
    }
    
    // Bounds checking: start and end must be within [0, base_len]
    if (start >= base_len) {
        IndexError("Slice start index out of bounds");
    }
    if (end > base_len) {
        IndexError("Slice end index out of bounds");
    }
    
    // Create slice: pointer to start position, length is end - start
    result.ptr = base + start;
    result.len = end - start;
    return result;
}