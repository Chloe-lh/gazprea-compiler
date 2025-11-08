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
  std::any astAny = builder.visitFile(tree);
  std::shared_ptr<FileNode> ast;
  // visitFile returns node_any (ASTNode*), need to cast via ASTNode first
  auto astNode = std::any_cast<std::shared_ptr<ASTNode>>(astAny);
  ast = std::dynamic_pointer_cast<FileNode>(astNode);

  // Print the AST
  std::cout << "--- Abstract Syntax Tree ---" << std::endl;
  gazprea::ASTPrinter printer(std::cout, true);
  ast->accept(printer);
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
 
  backend.lowerDialects();
  backend.dumpLLVM(os);
  

  return 0;
}
