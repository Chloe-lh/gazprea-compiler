/*
  This file provides the project ASTBuilder header using the ANTLR-generated
  visitor interface. The original repository contained two similarly named
  headers (`ASTBuilder.h` and `ASTbuilder.h`) with conflicting declarations.
  Replace the incorrect one with the correct, namespaced version so the
  generated `GazpreaParser`/`GazpreaBaseVisitor` types match the overrides.
*/

#pragma once

#include "antlr4-runtime.h"
#include "GazpreaBaseVisitor.h"
#include "GazpreaParser.h"
#include <any>

namespace gazprea {

class ASTBuilder : public GazpreaBaseVisitor {
    public:
        // Override all visit methods from GazpreaBaseVisitor
        std::any visitFile(GazpreaParser::FileContext *ctx) override;
        std::any visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx) override;
        std::any visitFunctionBlockTupleReturn(GazpreaParser::FunctionBlockTupleReturnContext *ctx) override;
        std::any visitFunctionStat(GazpreaParser::FunctionStatContext *ctx) override;
        std::any visitFunctionPrototype(GazpreaParser::FunctionPrototypeContext *ctx) override;
        std::any visitFunctionPrototypeTupleReturn(GazpreaParser::FunctionPrototypeTupleReturnContext *ctx) override;
        std::any visitProcedure(GazpreaParser::ProcedureContext *ctx) override;
        std::any visitExplicitTypedDec(GazpreaParser::ExplicitTypedDecContext *ctx) override;
        std::any visitInferredTypeDec(GazpreaParser::InferredTypeDecContext *ctx) override;
        std::any visitTupleTypedDec(GazpreaParser::TupleTypedDecContext *ctx) override;
        std::any visitAssignStat(GazpreaParser::AssignStatContext *ctx) override;
        std::any visitOutputStat(GazpreaParser::OutputStatContext *ctx) override;
        std::any visitInputStat(GazpreaParser::InputStatContext *ctx) override;
        std::any visitBreakStat(GazpreaParser::BreakStatContext *ctx) override;
        std::any visitContinueStat(GazpreaParser::ContinueStatContext *ctx) override;
        std::any visitReturnStat(GazpreaParser::ReturnStatContext *ctx) override;
        std::any visitCallStat(GazpreaParser::CallStatContext *ctx) override;
        std::any visitType(GazpreaParser::TypeContext *ctx) override;
        std::any visitBasicTypeAlias(GazpreaParser::BasicTypeAliasContext *ctx) override;
        std::any visitTupleTypeAlias(GazpreaParser::TupleTypeAliasContext *ctx) override;
        std::any visitAndExpr(GazpreaParser::AndExprContext *ctx) override;
        std::any visitTrueExpr(GazpreaParser::TrueExprContext *ctx) override;
        std::any visitIdExpr(GazpreaParser::IdExprContext *ctx) override;
        std::any visitMultExpr(GazpreaParser::MultExprContext *ctx) override;
        std::any visitAddExpr(GazpreaParser::AddExprContext *ctx) override;
        std::any visitCompExpr(GazpreaParser::CompExprContext *ctx) override;
        std::any visitExpExpr(GazpreaParser::ExpExprContext *ctx) override;
        std::any visitUnaryExpr(GazpreaParser::UnaryExprContext *ctx) override;
        std::any visitTupleTypeCastExpr(GazpreaParser::TupleTypeCastExprContext *ctx) override;
        std::any visitOrExpr(GazpreaParser::OrExprContext *ctx) override;
        std::any visitFalseExpr(GazpreaParser::FalseExprContext *ctx) override;
        std::any visitCharExpr(GazpreaParser::CharExprContext *ctx) override;
        std::any visitTupleAccessExpr(GazpreaParser::TupleAccessExprContext *ctx) override;
        std::any visitTupleLitExpr(GazpreaParser::TupleLitExprContext *ctx) override;
        std::any visitEqExpr(GazpreaParser::EqExprContext *ctx) override;
        std::any visitNotExpr(GazpreaParser::NotExprContext *ctx) override;
        std::any visitIntExpr(GazpreaParser::IntExprContext *ctx) override;
        std::any visitParenExpr(GazpreaParser::ParenExprContext *ctx) override;
        std::any visitRealExpr(GazpreaParser::RealExprContext *ctx) override;
        std::any visitTypeCastExpr(GazpreaParser::TypeCastExprContext *ctx) override;
        std::any visitFuncCallExpr(GazpreaParser::FuncCallExprContext *ctx) override;
        std::any visitTuple_dec(GazpreaParser::Tuple_decContext *ctx) override;
        std::any visitTuple_literal(GazpreaParser::Tuple_literalContext *ctx) override;
        std::any visitTuple_access(GazpreaParser::Tuple_accessContext *ctx) override;
        std::any visitBlock(GazpreaParser::BlockContext *ctx) override;
        std::any visitIf(GazpreaParser::IfContext *ctx) override;
        std::any visitLoopDefault(GazpreaParser::LoopDefaultContext *ctx) override;
        std::any visitWhileLoopBlock(GazpreaParser::WhileLoopBlockContext *ctx) override;
        std::any visitQualifier(GazpreaParser::QualifierContext *ctx) override;
        std::any visitReal(GazpreaParser::RealContext *ctx) override;
    };

} // namespace gazprea
