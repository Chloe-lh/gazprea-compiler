#pragma once
#include "AST.h"

class FuncCallExpr;

class ASTVisitor {
public:
  // Root
  virtual void visit(FileNode *node) = 0;

  // Functions
  virtual void visit(FuncStatNode *node) = 0;
  virtual void visit(FuncPrototypeNode *node) = 0;
  virtual void visit(FuncBlockNode *node) = 0;
  virtual void visit(ProcedureBlockNode *node) = 0;
  virtual void visit(ProcedurePrototypeNode *node) = 0;

  // Declarations
  virtual void visit(TypedDecNode *node) = 0;
  virtual void visit(InferredDecNode *node) = 0;
  virtual void visit(TupleTypedDecNode *node) = 0;
  virtual void visit(TypeAliasDecNode *node) = 0;
  virtual void visit(TypeAliasNode *node) = 0;
  virtual void visit(TupleTypeAliasNode *node) = 0;

  // Statements
  virtual void visit(AssignStatNode *node) = 0;
  virtual void visit(DestructAssignStatNode *node) = 0;
  virtual void visit(TupleAccessAssignStatNode *node) = 0;
  virtual void visit(OutputStatNode *node) = 0;
  virtual void visit(InputStatNode *node) = 0;
  virtual void visit(BreakStatNode *node) = 0;
  virtual void visit(ContinueStatNode *node) = 0;
  virtual void visit(ReturnStatNode *node) = 0;
  virtual void visit(CallStatNode *node) = 0;
  virtual void visit(IfNode *node) = 0;
  virtual void visit(LoopNode *node) = 0;
  virtual void visit(BlockNode *node) = 0;

  // Expressions
  virtual void visit(ParenExpr *node) = 0;
  virtual void visit(FuncCallExpr *node) = 0;
  // Backwards-compatible overload: if we get a CallExprNode, forward to
  // FuncCallExpr handler
  virtual void visit(CallExprNode *node) {
    visit(static_cast<FuncCallExpr *>(node));
  }
  virtual void visit(UnaryExpr *node) = 0;
  virtual void visit(ExpExpr *node) = 0;
  virtual void visit(MultExpr *node) = 0;
  virtual void visit(AddExpr *node) = 0;
  virtual void visit(CompExpr *node) = 0;
  virtual void visit(NotExpr *node) = 0;
  virtual void visit(EqExpr *node) = 0;
  virtual void visit(AndExpr *node) = 0;
  virtual void visit(OrExpr *node) = 0;
  virtual void visit(TrueNode *node) = 0;
  virtual void visit(FalseNode *node) = 0;
  virtual void visit(CharNode *node) = 0;
  virtual void visit(IntNode *node) = 0;
  virtual void visit(IdNode *node) = 0;
  virtual void visit(TupleLiteralNode *node) = 0;
  virtual void visit(TupleAccessNode *node) = 0;
  virtual void visit(TypeCastNode *node) = 0;
  virtual void visit(TupleTypeCastNode *node) = 0;
  virtual void visit(RealNode *node) = 0;
  virtual void visit(StringNode *node) = 0;
  virtual void visit(ArrayStrideExpr *node) = 0;
  virtual void visit(ArraySliceExpr *node)= 0;
  virtual void visit(ArrayAccessExpr *node) = 0;
  virtual void visit(ArrayInitNode *node) = 0;
  virtual void visit(ArrayDecNode *node) = 0;
  virtual void visit(ArrayTypeNode *node) =0;
  virtual void visit(ExprListNode *node) = 0;
  virtual void visit(ArrayLiteralNode *node) = 0;
  virtual void visit(RangeExprNode *node) = 0;
};
