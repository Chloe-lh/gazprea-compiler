#include "ASTBuilder.h"
#include "AST.h"
#include "antlr4-runtime.h"
#include <memory>
#include <stdexcept>

using namespace gazprea;
using namespace antlr4;

std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
    std::vector<std::shared_ptr<ASTNode>> statements;
    for (auto statCtx : ctx->stat()) {
        // Ensure we dispatch into the concrete rule within 'stat' and not the trailing ';'
        std::shared_ptr<ASTNode> stmt = std::any_cast<std::shared_ptr<ASTNode>>(visitStat(statCtx));
        if (stmt) {
            statements.push_back(std::move(stmt));
        }
    }
    return std::make_shared<FileNode>(std::move(statements));
}

std::any ASTBuilder::visitStat(GazpreaParser::StatContext *ctx) {
    // Route to the actual statement inside 'stat: <rule> END'
    if (ctx->intDec())       return visitIntDec(ctx->intDec());
    if (ctx->vectorDec())    return visitVectorDec(ctx->vectorDec());
    if (ctx->assign())       return visitAssign(ctx->assign());
    if (ctx->cond())         return visitCond(ctx->cond());
    if (ctx->loop())         return visitLoop(ctx->loop());
    if (ctx->print())        return visitPrint(ctx->print());
    return std::shared_ptr<ASTNode>{};
}

std::any ASTBuilder::visitExpr(GazpreaParser::ExprContext *ctx) {
    return visit(ctx->equalityExpr());
}

std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
    auto exprAny = visit(ctx->expr());
    auto expr = std::any_cast<std::shared_ptr<ExprNode>>(exprAny);
    auto node = std::make_shared<AssignNode>(id, expr);
    return std::static_pointer_cast<ASTNode>(node);
}

std::any ASTBuilder::visitIntDec(GazpreaParser::IntDecContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
    auto exprAny = visit(ctx->expr());
    auto expr = std::any_cast<std::shared_ptr<ExprNode>>(exprAny);
    auto node = std::make_shared<IntDecNode>(id, expr);
    return std::static_pointer_cast<ASTNode>(node);
}

std::any ASTBuilder::visitVectorDec(GazpreaParser::VectorDecContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
    auto exprAny = visit(ctx->expr());
    auto expr = std::any_cast<std::shared_ptr<ExprNode>>(exprAny);
    auto node = std::make_shared<VectorDecNode>(id, expr);
    return std::static_pointer_cast<ASTNode>(node);
}

std::any ASTBuilder::visitPrint(GazpreaParser::PrintContext *ctx) {
    auto exprAny = visit(ctx->expr());
    auto expr = std::any_cast<std::shared_ptr<ExprNode>>(exprAny);
    auto node = std::make_shared<PrintNode>(expr);
    return std::static_pointer_cast<ASTNode>(node);
}

std::any ASTBuilder::visitCond(GazpreaParser::CondContext *ctx) {
    auto condAny = visit(ctx->expr());
    auto condExpr = std::any_cast<std::shared_ptr<ExprNode>>(condAny);
    std::vector<std::shared_ptr<ASTNode>> body;
    for (auto statCtx : ctx->blockStat()) {
        std::shared_ptr<ASTNode> stmt = std::any_cast<std::shared_ptr<ASTNode>>(visitBlockStat(statCtx));
        if (stmt) {
            body.push_back(stmt);
        }
    }
    auto node = std::make_shared<CondNode>(condExpr, std::move(body));
    return std::static_pointer_cast<ASTNode>(node);
}

std::any ASTBuilder::visitLoop(GazpreaParser::LoopContext *ctx) {
    auto condAny = visit(ctx->expr());
    auto condExpr = std::any_cast<std::shared_ptr<ExprNode>>(condAny);
    std::vector<std::shared_ptr<ASTNode>> body;
    for (auto statCtx : ctx->blockStat()) {
        std::shared_ptr<ASTNode> stmt = std::any_cast<std::shared_ptr<ASTNode>>(visitBlockStat(statCtx));
        if (stmt) {
            body.push_back(std::move(stmt));
        }
    }
    auto node = std::make_shared<LoopNode>(condExpr, std::move(body));
    return std::static_pointer_cast<ASTNode>(node);
}

std::any ASTBuilder::visitBlockStat(GazpreaParser::BlockStatContext *ctx) {
    // Route to the actual statement inside 'blockStat: <rule> END'
    if (ctx->assign())   return visitAssign(ctx->assign());
    if (ctx->cond())     return visitCond(ctx->cond());
    if (ctx->loop())     return visitLoop(ctx->loop());
    if (ctx->print())    return visitPrint(ctx->print());
    return std::shared_ptr<ASTNode>{};
}

std::any ASTBuilder::visitEqualityExpr(GazpreaParser::EqualityExprContext *ctx) {
    size_t n = ctx->comparisonExpr().size();
    std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitComparisonExpr(ctx->comparisonExpr(0)));
    for (size_t i = 1; i < n; ++i) {
        // Find which operator was used at this position (i-1)
        std::string op;
        if (ctx->EQEQ(i-1)) {
            op = ctx->EQEQ(i-1)->getText();
        } else if (ctx->NEQ(i-1)) {
            op = ctx->NEQ(i-1)->getText();
        }
        auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitComparisonExpr(ctx->comparisonExpr(i)));
        node = std::make_shared<BinaryOpNode>(node, right, op);
    }
    return node;
}

std::any ASTBuilder::visitComparisonExpr(GazpreaParser::ComparisonExprContext *ctx) {
    size_t n = ctx->addSubExpr().size();
    std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitAddSubExpr(ctx->addSubExpr(0)));
    for (size_t i = 1; i < n; ++i) {
        std::string op;
        if (ctx->LT(i-1)) {
            op = ctx->LT(i-1)->getText();
        } else if (ctx->GT(i-1)) {
            op = ctx->GT(i-1)->getText();
        }
        auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitAddSubExpr(ctx->addSubExpr(i)));
        node = std::make_shared<BinaryOpNode>(node, right, op);
    }
    return node;
}

std::any ASTBuilder::visitAddSubExpr(GazpreaParser::AddSubExprContext *ctx) {
    size_t n = ctx->mulDivExpr().size();
    std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitMulDivExpr(ctx->mulDivExpr(0)));
    for (size_t i = 1; i < n; ++i) {
        std::string op;
        if (ctx->ADD(i-1)) {
            op = ctx->ADD(i-1)->getText();
        } else if (ctx->MINUS(i-1)) {
            op = ctx->MINUS(i-1)->getText();
        }
        auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitMulDivExpr(ctx->mulDivExpr(i)));
        node = std::make_shared<BinaryOpNode>(node, right, op);
    }
    return node;
}

std::any ASTBuilder::visitMulDivExpr(GazpreaParser::MulDivExprContext *ctx) {
    size_t n = ctx->rangeExpr().size();
    std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitRangeExpr(ctx->rangeExpr(0)));
    for (size_t i = 1; i < n; ++i) {
        std::string op;
        if (ctx->MULT(i-1)) {
            op = ctx->MULT(i-1)->getText();
        } else if (ctx->DIV(i-1)) {
            op = ctx->DIV(i-1)->getText();
        }
        auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitRangeExpr(ctx->rangeExpr(i)));
        node = std::make_shared<BinaryOpNode>(node, right, op);
    }
    return node;
}

std::any ASTBuilder::visitRangeExpr(GazpreaParser::RangeExprContext *ctx) {
    if (ctx->indexExpr().size() == 1) {
        // If there's only one indexExpr and no '..', return it directly (do not wrap in RangeNode)
        return visitIndexExpr(ctx->indexExpr(0));
    }
    auto startAny = visitIndexExpr(ctx->indexExpr(0));
    auto endAny = visitIndexExpr(ctx->indexExpr(1));
    auto node = std::make_shared<RangeNode>(
        std::any_cast<std::shared_ptr<ExprNode>>(startAny),
        std::any_cast<std::shared_ptr<ExprNode>>(endAny)
    );
    return std::static_pointer_cast<ExprNode>(node);
}

std::any ASTBuilder::visitIndexExpr(GazpreaParser::IndexExprContext *ctx) {
    // no expr, go to atom
    if (!ctx->expr()) {
        return visit(ctx->atom());
    }

    auto arrayAny = visit(ctx->atom());
    auto array = std::any_cast<std::shared_ptr<ExprNode>>(arrayAny);
    auto index = std::any_cast<std::shared_ptr<ExprNode>>(visit(ctx->expr()));
    auto node = std::make_shared<IndexNode>(array, index);
    return std::static_pointer_cast<ExprNode>(node);
}

std::any ASTBuilder::visitGenerator(GazpreaParser::GeneratorContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
    auto domAny = visit(ctx->expr(0));
    auto bodyAny = visit(ctx->expr(1));
    auto node = std::make_shared<GeneratorNode>(
        id,
        std::any_cast<std::shared_ptr<ExprNode>>(domAny),
        std::any_cast<std::shared_ptr<ExprNode>>(bodyAny));
    return std::static_pointer_cast<ExprNode>(node);
}

std::any ASTBuilder::visitFilter(GazpreaParser::FilterContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
    auto domAny = visit(ctx->expr(0));
    auto predAny = visit(ctx->expr(1));
    auto node = std::make_shared<FilterNode>(
        id,
        std::any_cast<std::shared_ptr<ExprNode>>(domAny),
        std::any_cast<std::shared_ptr<ExprNode>>(predAny));
    return std::static_pointer_cast<ExprNode>(node);
}

std::any ASTBuilder::visitAtom(GazpreaParser::AtomContext *ctx) {
    if (ctx->INT()) {
        const std::string literal = ctx->INT()->getText();
        try {
            size_t parsedChars = 0;
            int value = std::stoi(literal, &parsedChars, 10);
            if (parsedChars != literal.size()) {
                throw std::runtime_error("TypeError: Integer literal '" + literal + "' is not a valid signed int.");
            }
            auto node = std::make_shared<IntNode>(value);
            return std::static_pointer_cast<ExprNode>(node);
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("TypeError: Integer literal '" + literal + "' is not a valid signed int.");
        } catch (const std::out_of_range&) {
            throw std::runtime_error("RangeError: Integer literal '" + literal + "' is out of range for signed int.");
        }
    } else if (ctx->ID()) {
        std::string name = ctx->ID()->getText();
        auto node = std::make_shared<IdNode>(name);
        return std::static_pointer_cast<ExprNode>(node);
    } else if (ctx->generator()) {
        return visitGenerator(ctx->generator());
    } else if (ctx->filter()) {
        return visitFilter(ctx->filter());
    } else if (ctx->expr()) {
        return visit(ctx->expr());
    }
    return std::shared_ptr<ExprNode>{};
}
