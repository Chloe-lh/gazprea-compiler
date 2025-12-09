#include "CompileTimeExceptions.h"
#include "GazpreaLexer.h"
#include "GazpreaParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"
#include "tree/ParseTreeWalker.h"

#include "BackEnd.h"
#include "ASTBuilder.h"
#include "AST.h"
#include "ASTPrinter.h"
#include "ErrorListener.h"
#include "SemanticAnalysisVisitor.h"
#include "ConstantFolding.h"
#include "MLIRgen.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

bool VERBOSE_ERRORS = false;
bool DUMP_AST_POST_SEMA = false;

int main(int argc, char **argv) {
  std::vector<std::string> positional_args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--verbose-errors" || arg == "-ve") {
      VERBOSE_ERRORS = true;
    } else if (arg == "--dump-ast-post-sema" || arg == "-daps") {
      DUMP_AST_POST_SEMA = true;
    } else {
      positional_args.push_back(arg);
    }
  }

  if (positional_args.size() < 2) {
    std::cout << "Missing required argument.\n"
              << "Required arguments: <input file path> <output file path>\n"
              << "Optional flags: --verbose-errors --dump-ast-post-sema\n";
    return 1;
  }

  const std::string &input_path = positional_args[0];
  const std::string &output_path = positional_args[1];

  // Open the file then parse and lex it.
  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(input_path);
  gazprea::GazpreaLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  gazprea::GazpreaParser parser(&tokens);

  // Add our custom error listener for syntax errors
  parser.removeErrorListeners();
  parser.addErrorListener(new ErrorListener());

  std::shared_ptr<FileNode> ast;
  try {
      // Get the root of the parse tree. Use your base rule name.
      auto *tree = parser.file();
      gazprea::ASTBuilder builder;

      std::any astAny = builder.visitFile(tree);
      // visitFile returns node_any (ASTNode*), need to cast via ASTNode first
      auto astNode = std::any_cast<std::shared_ptr<ASTNode>>(astAny);
      ast = std::dynamic_pointer_cast<FileNode>(astNode);
  } catch (const CompileTimeException &e) {
      std::cerr << e.what();
      return 1;
  } catch (const std::exception &e) {
      std::cerr << e.what();
      // std::cerr << "Internal error building AST ("
      //           << typeid(e).name() << "): " << e.what() << '\n';
      return 1;
  } catch (...) {
      return 1;
  }
  
  if (!ast) {
    std::cerr << "Failed to build AST: got null FileNode" << std::endl;
    return 1;
  }

  // Print the AST
  std::cout << "--- Abstract Syntax Tree ---" << std::endl;
  try {
    gazprea::ASTPrinter printer(std::cout, true);
    ast->accept(printer);
  } catch (const std::exception& e) {
    std::cerr << "Error printing AST: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown error printing AST" << std::endl;
    return 1;
  }
  std::cout << "--------------------------\n" << std::endl;

  // HOW TO USE A VISITOR
  // Make the visitor
  // MyVisitor visitor;
  // Visit the tree
  // visitor.visit(tree);
  try{
    SemanticAnalysisVisitor semVisitor;
    ast->accept(semVisitor);
    Scope* rootScope = semVisitor.getRootScope();
    const auto* scopeMap = &semVisitor.getScopeMap();

    if (DUMP_AST_POST_SEMA) {
      std::cout << "--- Abstract Syntax Tree (post-sema) ---" << std::endl;
      gazprea::ASTPrinter printer(std::cout, true);
      ast->accept(printer);
      std::cout << "--------------------------\n" << std::endl;
    }

    // Run constant folding pass (uses semantic info from previous pass)
    ConstantFoldingVisitor cfv;
    ast->accept(cfv);
    std::ofstream os(output_path);
    BackEnd backend;
    
    // backend.emitModule(); demo module
    MLIRGen mlirGen(backend, rootScope, scopeMap);
    ast->accept(mlirGen);

  
    // Debug
    backend.dumpMLIR(std::cout);

    if (backend.lowerDialects() != 0) {
      std::cerr << "Lowering failed; aborting translation.";
      return 1;
    }
    backend.dumpLLVM(os);
  }catch (CompileTimeException &e){
    std::cerr << e.what();
    // std::cerr << e.what() << std::endl;
    return 1;

  }

  return 0;
}
