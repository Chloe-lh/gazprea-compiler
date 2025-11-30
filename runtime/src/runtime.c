#include <stdio.h>
#include <stdbool.h>
#include "../include/run_time_errors.h"

// Add function named dummyPrint with signature void(int) to llvm to have this linked in.
void dummyPrint(int i) {
  printf("I'm a function! %d\n", i);
}

// Functions for reading from std_input
/*
int readInt(int* i) {
  int c;

  // skip whitespace
  while ((c = getchar()) == ' ' || c == '\n' || c == '\t') {}

  // read integer
  int sign = 1;
  if (c == '-') {
      sign = -1;
      c = getchar();
  }

  int value = 0;
  while (c >= '0' && c <= '9') {
      value = value * 10 + (c - '0');
      c = getchar();
  }

  return sign * value;
}
  */

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
