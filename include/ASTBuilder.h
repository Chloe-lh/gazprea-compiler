#pragma once
#include "GazpreaBaseVisitor.h" 
#include "GazpreaParser.h"       
#include "antlr4-runtime.h"
#include <any>

using namespace gazprea;
/*
Class converts Parse tree produced by ANTLR into AST Tree
*/
class ASTBuilder: public GazpreaBaseVisitor{
    public:
        // Top-level statements and blocks
        std::any visitFile(GazpreaParser::FileContext *ctx) override;
        std::any visitStat(GazpreaParser::StatContext *ctx) override;
        std::any visitBlockStat(GazpreaParser::BlockStatContext *ctx) override;
        std::any visitExpr(GazpreaParser::ExprContext *ctx) override;
        std::any visitIntDec(GazpreaParser::IntDecContext *ctx) override;
        std::any visitVectorDec(GazpreaParser::VectorDecContext *ctx) override;
        std::any visitAssign(GazpreaParser::AssignContext *ctx) override;
        std::any visitPrint(GazpreaParser::PrintContext *ctx) override;
        std::any visitCond(GazpreaParser::CondContext *ctx) override;
        std::any visitLoop(GazpreaParser::LoopContext *ctx) override;
        
        std::any visitEqualityExpr(GazpreaParser::EqualityExprContext *ctx) override;
        std::any visitComparisonExpr(GazpreaParser::ComparisonExprContext *ctx) override;
        std::any visitAddSubExpr(GazpreaParser::AddSubExprContext *ctx) override;
        std::any visitMulDivExpr(GazpreaParser::MulDivExprContext *ctx) override;
        std::any visitRangeExpr(GazpreaParser::RangeExprContext *ctx) override;
        std::any visitIndexExpr(GazpreaParser::IndexExprContext *ctx) override;
        
        std::any visitGenerator(GazpreaParser::GeneratorContext *ctx) override;
        std::any visitFilter(GazpreaParser::FilterContext *ctx) override;
        std::any visitAtom(GazpreaParser::AtomContext *ctx) override;
    
};
