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

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Missing required argument.\n"
              << "Required arguments: <input file path> <output file path>\n";
    return 1;
  }

  // Open the file then parse and lex it.
  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(argv[1]);
  gazprea::GazpreaLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  gazprea::GazpreaParser parser(&tokens);

  // Add our custom error listener for syntax errors
  parser.removeErrorListeners();
  parser.addErrorListener(new ErrorListener()); 

  // Get the root of the parse tree. Use your base rule name.
  auto *tree = parser.file();
  gazprea::ASTBuilder builder;
  
  std::shared_ptr<FileNode> ast;
  try {
    std::any astAny = builder.visitFile(tree);
    // visitFile returns node_any (ASTNode*), need to cast via ASTNode first
    auto astNode = std::any_cast<std::shared_ptr<ASTNode>>(astAny);
    ast = std::dynamic_pointer_cast<FileNode>(astNode);
  } catch (const std::exception& e) {
    std::cerr << "Error building AST: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown error building AST" << std::endl;
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

  SemanticAnalysisVisitor semVisitor;
  ast->accept(semVisitor);
  Scope* rootScope = semVisitor.getRootScope();
  const auto* scopeMap = &semVisitor.getScopeMap();

  // Run constant folding pass (uses semantic info from previous pass)
  ConstantFoldingVisitor cfv;
  ast->accept(cfv);


  std::ofstream os(argv[2]);
  BackEnd backend;
  
  // backend.emitModule(); demo module
  MLIRGen mlirGen(backend, rootScope, scopeMap);
  ast->accept(mlirGen);
 
  // Debug
  backend.dumpMLIR(std::cerr);

  if (backend.lowerDialects() != 0) {
    std::cerr << "Lowering failed; aborting translation." << std::endl;
    return 1;
  }
  backend.dumpLLVM(os);
  

  return 0;
}
