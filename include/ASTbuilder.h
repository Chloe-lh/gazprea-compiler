#include "antlr4-runtime.h"
#include "GazpreaParser.h"
#include "GazpreaBaseVisitor.h"


class ASTVisitor; //foward reference

class ASTBuilder : public GazpreaBaseVisitor {
public:
    // File and top-level
    std::any visitFile(GazpreaParser::FileContext *ctx) override;

    // Declarations
    std::any visitExplicitTypedDec(GazpreaParser::ExplicitTypedDecContext *ctx) override;
    std::any visitInferredTypeDec(GazpreaParser::InferredTypeDecContext *ctx) override;
    std::any visitTupleTypedDec(GazpreaParser::TupleTypedDecContext *ctx) override;

    // Functions and procedures
    std::any visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx) override;
    std::any visitFunctionBlockTupleReturn(GazpreaParser::FunctionBlockTupleReturnContext *ctx) override;
    std::any visitFunctionStat(GazpreaParser::FunctionStatContext *ctx) override;
    std::any visitFunctionPrototype(GazpreaParser::FunctionPrototypeContext *ctx) override;
    std::any visitFunctionPrototypeTupleReturn(GazpreaParser::FunctionPrototypeTupleReturnContext *ctx) override;
    std::any visitProcedure(GazpreaParser::ProcedureContext *ctx) override;

    // Type aliases
    std::any visitBasicTypeAlias(GazpreaParser::BasicTypeAliasContext *ctx) override;
    std::any visitTupleTypeAlias(GazpreaParser::TupleTypeAliasContext *ctx) override;

    // Statements
    std::any visitAssignStat(GazpreaParser::AssignStatContext *ctx) override;
    std::any visitOutputStat(GazpreaParser::OutputStatContext *ctx) override;
    std::any visitInputStat(GazpreaParser::InputStatContext *ctx) override;
    std::any visitBreakStat(GazpreaParser::BreakStatContext *ctx) override;
    std::any visitContinueStat(GazpreaParser::ContinueStatContext *ctx) override;
    std::any visitReturnStat(GazpreaParser::ReturnStatContext *ctx) override;
    std::any visitCallStat(GazpreaParser::CallStatContext *ctx) override;

    // Expressions
    std::any visitTupleAccessExpr(GazpreaParser::TupleAccessExprContext *ctx) override;
    std::any visitFuncCallExpr(GazpreaParser::FuncCallExprContext *ctx) override;
    std::any visitParenExpr(GazpreaParser::ParenExprContext *ctx) override;
    std::any visitUnaryExpr(GazpreaParser::UnaryExprContext *ctx) override;
    std::any visitExpExpr(GazpreaParser::ExpExprContext *ctx) override;
    std::any visitMultExpr(GazpreaParser::MultExprContext *ctx) override;
    std::any visitAddExpr(GazpreaParser::AddExprContext *ctx) override;
    std::any visitCompExpr(GazpreaParser::CompExprContext *ctx) override;
    std::any visitNotExpr(GazpreaParser::NotExprContext *ctx) override;
    std::any visitEqExpr(GazpreaParser::EqExprContext *ctx) override;
    std::any visitAndExpr(GazpreaParser::AndExprContext *ctx) override;
    std::any visitOrExpr(GazpreaParser::OrExprContext *ctx) override;
    std::any visitTrueExpr(GazpreaParser::TrueExprContext *ctx) override;
    std::any visitFalseExpr(GazpreaParser::FalseExprContext *ctx) override;
    std::any visitCharExpr(GazpreaParser::CharExprContext *ctx) override;
    std::any visitIntExpr(GazpreaParser::IntExprContext *ctx) override;
    std::any visitRealExpr(GazpreaParser::RealExprContext *ctx) override;
    std::any visitTupleLitExpr(GazpreaParser::TupleLitExprContext *ctx) override;
    std::any visitTypeCastExpr(GazpreaParser::TypeCastExprContext *ctx) override;
    std::any visitTupleTypeCastExpr(GazpreaParser::TupleTypeCastExprContext *ctx) override;
    std::any visitIdExpr(GazpreaParser::IdExprContext *ctx) override;

    // Blocks, if, loop
    std::any visitBlock(GazpreaParser::BlockContext *ctx) override;
    std::any visitIf(GazpreaParser::IfContext *ctx) override;
    std::any visitLoop(GazpreaParser::LoopContext *ctx) override;

    // Qualifier, type, etc.
    std::any visitQualifier(GazpreaParser::QualifierContext *ctx) override;
    std::any visitType(GazpreaParser::TypeContext *ctx) override;
};