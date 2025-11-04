#include "GazpreaLexer.h"
#include "GazpreaParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"
#include "tree/ParseTreeWalker.h"

#include "BackEnd.h"
#include "ASTBuilder.h"
#include "AST.h"
#include "ErrorListener.h"

#include <iostream>
#include <fstream>

int main(int argc, char **argv) {
  // if (argc < 3) {
  //   std::cout << "Missing required argument.\n"
  //             << "Required arguments: <input file path> <output file path>\n";
  //   return 1;
  // }

   const std::string src = R"(
        dec var integer x = 1 END
        dec var integer y = 2 END
        x = x + y END
    )";

   
   

  // Open the file then parse and lex it.
  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(argv[1]);
  gazprea::GazpreaLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  gazprea::GazpreaParser parser(&tokens);

  // Add our custom error listener for syntax errors
  // parser.removeErrorListeners();
  // parser.addErrorListener(new ErrorListener()); 

  // Get the root of the parse tree. Use your base rule name.
  
  auto fileCtx = parser.file();
  gazprea::ASTBuilder builder;
  std::any anyAst = builder.visitFile(fileCtx);

  assert(anyAst.has_value());
  auto fileNode = std::any_cast<std::shared_ptr<ASTNode>>(anyAst);
  assert(fileNode);

    // Ensure it's a FileNode
  auto fnode = std::dynamic_pointer_cast<FileNode>(fileNode);
  assert(fnode);

  // HOW TO USE A VISITOR
  // Make the visitor
  // MyVisitor visitor;
  // Visit the tree
  // visitor.visit(tree);


  if (fnode->stats.empty()) {
      std::cerr << "ASTBuilder produced empty file node\n";
      return 2;
  }

  std::cout << "ASTBuilder smoke test passed: top-level node count = " << fnode->stats.size() << "\n";
  return 0;

  // std::ofstream os(argv[2]);
  // BackEnd backend;
  // backend.emitModule();
  // backend.lowerDialects();
  // backend.dumpLLVM(os);

  // return 0;
}
