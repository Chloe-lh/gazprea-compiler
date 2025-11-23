#pragma once
#include "AST.h"
#include "GazpreaBaseVisitor.h"
#include "GazpreaParser.h"
#include <any>

/*
Class converts Parse tree produced by ANTLR into AST Tree
*/

namespace gazprea {
class ASTBuilder : public GazpreaBaseVisitor {
public:
static void setLocationFromCtx(std::shared_ptr<ASTNode> node, antlr4::ParserRuleContext *ctx);
  // Top-level statements and blocks (implemented in src/ASTBuilder.cpp)
  std::any visitFile(gazprea::GazpreaParser::FileContext *ctx) override;
  std::any visitBlock(gazprea::GazpreaParser::BlockContext *ctx) override;

  // Declarations
  std::any visitExplicitTypedDec(
      gazprea::GazpreaParser::ExplicitTypedDecContext *ctx) override;
  std::any visitInferredTypeDec(
      gazprea::GazpreaParser::InferredTypeDecContext *ctx) override;
  std::any visitTupleTypedDec(
      gazprea::GazpreaParser::TupleTypedDecContext *ctx) override;

  // Types / aliases / qualifiers
  std::any visitTupleTypeAlias(
      gazprea::GazpreaParser::TupleTypeAliasContext *ctx) override;
  std::any visitBasicTypeAlias(
      gazprea::GazpreaParser::BasicTypeAliasContext *ctx) override;
  std::any visitType(gazprea::GazpreaParser::TypeContext *ctx) override;
  std::any
  visitQualifier(gazprea::GazpreaParser::QualifierContext *ctx) override;

  // Statements
  std::any
  visitAssignStat(gazprea::GazpreaParser::AssignStatContext *ctx) override;
  std::any visitDestructAssignStat(
      gazprea::GazpreaParser::DestructAssignStatContext *ctx) override;
  std::any visitTupleAccessAssignStat(
      gazprea::GazpreaParser::TupleAccessAssignStatContext *ctx) override;
  std::any
  visitBreakStat(gazprea::GazpreaParser::BreakStatContext *ctx) override;
  std::any
  visitContinueStat(gazprea::GazpreaParser::ContinueStatContext *ctx) override;
  std::any
  visitReturnStat(gazprea::GazpreaParser::ReturnStatContext *ctx) override;
  std::any visitCallStat(gazprea::GazpreaParser::CallStatContext *ctx) override;
  std::any
  visitInputStat(gazprea::GazpreaParser::InputStatContext *ctx) override;
  std::any
  visitOutputStat(gazprea::GazpreaParser::OutputStatContext *ctx) override;

  // Function / procedure related
  std::any visitFunctionBlock(
      gazprea::GazpreaParser::FunctionBlockContext *ctx) override;
  std::any visitFunctionBlockTupleReturn(
      gazprea::GazpreaParser::FunctionBlockTupleReturnContext *ctx) override;
  std::any visitFunctionPrototype(
      gazprea::GazpreaParser::FunctionPrototypeContext *ctx) override;
  std::any visitFunctionPrototypeTupleReturn(
      gazprea::GazpreaParser::FunctionPrototypeTupleReturnContext *ctx)
      override;
  std::any visitProcedurePrototype(
      gazprea::GazpreaParser::ProcedurePrototypeContext *ctx) override;
  std::any visitProcedureBlock(
      gazprea::GazpreaParser::ProcedureBlockContext *ctx) override;
  std::any
  visitFunctionStat(gazprea::GazpreaParser::FunctionStatContext *ctx) override;

  // Expressions (various grammar alternatives)
  std::any
  visitFuncCallExpr(gazprea::GazpreaParser::FuncCallExprContext *ctx) override;
  std::any
  visitParenExpr(gazprea::GazpreaParser::ParenExprContext *ctx) override;
  std::any
  visitUnaryExpr(gazprea::GazpreaParser::UnaryExprContext *ctx) override;
  std::any visitNotExpr(gazprea::GazpreaParser::NotExprContext *ctx) override;
  std::any visitCompExpr(gazprea::GazpreaParser::CompExprContext *ctx) override;
  std::any visitAddExpr(gazprea::GazpreaParser::AddExprContext *ctx) override;
  std::any visitMultExpr(gazprea::GazpreaParser::MultExprContext *ctx) override;
  std::any visitEqExpr(gazprea::GazpreaParser::EqExprContext *ctx) override;
  std::any visitAndExpr(gazprea::GazpreaParser::AndExprContext *ctx) override;
  std::any visitOrExpr(gazprea::GazpreaParser::OrExprContext *ctx) override;
  std::any
  visitTypeCastExpr(gazprea::GazpreaParser::TypeCastExprContext *ctx) override;

  // Literals / atoms
  std::any visitIntExpr(gazprea::GazpreaParser::IntExprContext *ctx) override;
  std::any visitIdExpr(gazprea::GazpreaParser::IdExprContext *ctx) override;
  std::any visitCharExpr(gazprea::GazpreaParser::CharExprContext *ctx) override;
  std::any visitRealExpr(gazprea::GazpreaParser::RealExprContext *ctx) override;
  std::any
  visitStringExpr(gazprea::GazpreaParser::StringExprContext *ctx) override;
  std::any visitTrueExpr(gazprea::GazpreaParser::TrueExprContext *ctx) override;
  std::any
  visitFalseExpr(gazprea::GazpreaParser::FalseExprContext *ctx) override;

  // Tuple support
  std::any visitTuple_literal(
      gazprea::GazpreaParser::Tuple_literalContext *ctx) override;
  std::any
  visitTupleLitExpr(gazprea::GazpreaParser::TupleLitExprContext *ctx) override;
  std::any
  visitTuple_dec(gazprea::GazpreaParser::Tuple_decContext *ctx) override;
  std::any visitTupleAccessExpr(
      gazprea::GazpreaParser::TupleAccessExprContext *ctx) override;
  std::any
  visitTupleTypeCastExpr(GazpreaParser::TupleTypeCastExprContext *ctx) override;
  // Control flow
  std::any visitIfStat(gazprea::GazpreaParser::IfStatContext *ctx) override;
  std::any visitLoopStat(gazprea::GazpreaParser::LoopStatContext *ctx) override;
  std::any visitWhileLoopBlock(
      gazprea::GazpreaParser::WhileLoopBlockContext *ctx) override;
  std::any
  visitLoopDefault(gazprea::GazpreaParser::LoopDefaultContext *ctx) override;
};
} // namespace gazprea
