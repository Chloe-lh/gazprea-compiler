#pragma once
#include "ASTVisitor.h"
#include "AST.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <optional>

class ConstantFoldingVisitor : public ASTVisitor {
public:
  ConstantFoldingVisitor() = default;

  // Root
  void visit(FileNode *node) override;

  // Functions
  void visit(FuncStatNode *node) override;
  void visit(FuncPrototypeNode *node) override;
  void visit(FuncBlockNode *node) override;
  void visit(ProcedureBlockNode *node) override;
  void visit(ProcedurePrototypeNode *node) override;
  void visit(BuiltInFuncNode *node) override;

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
  void visit(StructAccessAssignStatNode *node) override;
  void visit(ArrayAccessAssignStatNode *node) override;
  void visit(OutputStatNode *node) override;
  void visit(InputStatNode *node) override;
  void visit(BreakStatNode *node) override;
  void visit(ContinueStatNode *node) override;
  void visit(ReturnStatNode *node) override;
  void visit(CallStatNode *node) override;
  void visit(IfNode *node) override;
  void visit(LoopNode *node) override;
  void visit(IteratorLoopNode *node) override;
  void visit(GeneratorExprNode *node) override;
  void visit(BlockNode *node) override;

  // Expressions
  void visit(ParenExpr *node) override;
  void visit(FuncCallExprOrStructLiteral *node) override;
  void visit(UnaryExpr *node) override;
  void visit(ExpExpr *node) override;
  void visit(DotExpr *node) override;
  void visit(MultExpr *node) override;
  void visit(AddExpr *node) override;
  void visit(CompExpr *node) override;
  void visit(NotExpr *node) override;
  void visit(EqExpr *node) override;
  void visit(AndExpr *node) override;
  void visit(OrExpr *node) override;
  void visit(ConcatExpr *node) override;
  void visit(TrueNode *node) override;
  void visit(FalseNode *node) override;
  void visit(CharNode *node) override;
  void visit(IntNode *node) override;
  void visit(IdNode *node) override;
  void visit(StructAccessNode *node) override;
  void visit(TupleLiteralNode *node) override;
  void visit(TupleAccessNode *node) override;
  void visit(TypeCastNode *node) override;
  void visit(TupleTypeCastNode *node) override;
  void visit(RealNode *node) override;
  void visit(StringNode *node) override;

  void visit(ArrayStrideExpr *node) override;
  void visit(ArraySliceExpr *node)override;
  void visit(ArrayAccessNode *node) override;
  void visit(ArrayTypedDecNode *node) override;
  void visit(ExprListNode *node) override;
  void visit(ArrayLiteralNode *node) override;
  void visit(RangeExprNode *node) override;

private:
  // stack of lexical scopes mapping identifier -> ConstantValue
  std::vector<std::unordered_map<std::string, ConstantValue>> scopes_;

  void pushScope(); 
  void popScope();
  std::optional<ConstantValue> lookup(const std::string &ident) const;
  void setConstInCurrentScope(const std::string &ident, const ConstantValue &cv);
  void removeConst(const std::string &ident);
  static std::optional<std::vector<long double>> extract1D(const ArrayLiteralNode *arr); // convert array of nodes into array of numeric
  static std::optional<std::vector<std::vector<long double>>> extract2D(const ArrayLiteralNode *arr); //convert matric of nodes into matrix of numeric
  // Debug helper: print current scopes to stdout (used during development)
  void debugPrintScopes() const;
 
};
