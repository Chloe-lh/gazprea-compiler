# Gazprea Compiler
Contributors: Chloe Haynes, Jacob Yang, and Maksimus Andruchow

## Description
This is a compiler designed for IBM's coding language Gazprea. Our comiler translates Gazprea source code into intermediate representation (IR), performing lexical analysis, semantic checks, and error reporting. 

## Features
- **Lexical Analysis**: Tokenizes source code according to the Gazprea grammar.
- **Parsing**: Implements bottom‑up parsing to construct abstract syntax trees (ASTs).
- **Semantic Analysis**: Performs type checking, scope resolution, and detects semantic errors.
- **Intermediate Representation (IR)**: Generates a structured IR for further analysis or optimization.
- **Constant Folding**: Optimizes expressions by evaluating constant sub-expressions at compile time
- **Error Handling**: Provides clear, informative diagnostics for syntax and semantic issues.

# GazpreaBase
The base cmake setup for Gazprea assignment.

Author: Braedy Kuzma (braedy@ualberta.ca)  

Updated by: Deric Cheung (dacheung@ualberta.ca)

Updated by: Quinn Pham (qpham@ualberta.ca)

## Building
### Linux
  1. Install git, java (only the runtime is necessary), and cmake (>= v3.0).
     - Until now, cmake has found the dependencies without issues. If you
       encounter an issue, let a TA know and we can fix it.
  1. Make a directory that you intend to build the project in and change into
     that directory.
  1. Run `cmake <path-to-Gazprea-Base>`.
  1. Run `make`.
  1. Done.
