#pragma once

#include "ASTVisitor.h"
#include "AST.h"
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace gazprea {

/**
 * @class ASTPrinter
 * @brief An ASTVisitor implementation that prints the AST as a formatted tree.
 */
class ASTPrinter : public ASTVisitor {
public:
  ASTPrinter(std::ostream &output_stream = std::cout, bool enableColor = true);

  // Visitor methods overridden from ASTVisitor
  void visit(FileNode *node) override;

  // Functions
  void visit(FuncStatNode *node) override;
  void visit(FuncPrototypeNode *node) override;
  void visit(FuncBlockNode *node) override;
  void visit(ProcedureBlockNode *node) override;
  void visit(ProcedurePrototypeNode *node) override;

  // Declarations
  void visit(TypedDecNode *node) override;
  void visit(InferredDecNode *node) override;
  void visit(TupleTypedDecNode *node) override;
  void visit(StructTypedDecNode *node) override;
  void visit(TypeAliasDecNode *node) override;
  void visit(TypeAliasNode *node) override;
  void visit(TupleTypeAliasNode *node) override;

  // Statements
  void visit(AssignStatNode *node) override;
  void visit(DestructAssignStatNode *node) override;
  void visit(TupleAccessAssignStatNode *node) override;
  void visit(OutputStatNode *node) override;
  void visit(InputStatNode *node) override;
  void visit(BreakStatNode *node) override;
  void visit(ContinueStatNode *node) override;
  void visit(ReturnStatNode *node) override;
  void visit(CallStatNode *node) override;
  void visit(IfNode *node) override;
  void visit(LoopNode *node) override;
  void visit(BlockNode *node) override;

  // Expressions
  void visit(ParenExpr *node) override;
  void visit(FuncCallExpr *node) override;
  void visit(UnaryExpr *node) override;
  void visit(ExpExpr *node) override;
  void visit(MultExpr *node) override;
  void visit(AddExpr *node) override;
  void visit(CompExpr *node) override;
  void visit(NotExpr *node) override;
  void visit(EqExpr *node) override;
  void visit(AndExpr *node) override;
  void visit(OrExpr *node) override;
  void visit(TrueNode *node) override;
  void visit(FalseNode *node) override;
  void visit(CharNode *node) override;
  void visit(IntNode *node) override;
  void visit(IdNode *node) override;
  void visit(TupleLiteralNode *node) override;
  void visit(TupleAccessNode *node) override;
  void visit(TypeCastNode *node) override;
  void visit(TupleTypeCastNode *node) override;
  void visit(RealNode *node) override;
  void visit(StringNode *node) override;

  // Array & Range Expressions
  void visit(ArrayStrideExpr *node) override;
  void visit(ArraySliceExpr *node) override;
  void visit(ArrayAccessExpr *node) override;
  void visit(ArrayTypedDecNode *node) override;
  void visit(ArrayTypeNode *node) override;
  void visit(ExprListNode *node) override;
  void visit(ArrayLiteralNode *node) override;
  void visit(RangeExprNode *node) override;

  

private:
  std::ostream &out;
  int indent = 0;
  std::vector<bool> isLastChild;
  bool colorEnabled = true;
  std::vector<std::string> levelColors;

  void printTreeLine(const std::string &nodeType,
                     const std::string &details = "");
  void pushChildContext(bool isLast);
  void popChildContext();
};

} // namespace gazprea
