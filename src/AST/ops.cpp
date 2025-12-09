#include "AST.h"
#include "ConstantHelpers.h"
#include  "GazpreaParser.h"
#include "ASTBuilder.h"
#include "ASTBuilderUtils.h"


namespace gazprea{
    std::any ASTBuilder::visitUnaryExpr(GazpreaParser::UnaryExprContext *ctx) {
        std::shared_ptr<ExprNode> expr = nullptr;
        std::string op;
        if (ctx->ADD()) {
            op = ctx->ADD()->getText();
        } else if (ctx->MINUS()) {
            op = ctx->MINUS()->getText();
        } 
        
        if (ctx->expr()) {
            auto anyExpr = visit(ctx->expr());
            if (anyExpr.has_value()) {
            expr = safe_any_cast_ptr<ExprNode>(anyExpr);
            }
        }
        
            auto node = std::make_shared<UnaryExpr>(op, expr);
            setLocationFromCtx(node, ctx);
            node->constant.reset();
            if (expr && expr->constant) {
                auto res = gazprea::computeUnaryNumeric(*expr->constant, op);
                if (res) {
                node->constant = *res;
                node->type = node->constant->type;
                }
            }
            return expr_any(std::move(node));
            }
            std::any ASTBuilder::visitNotExpr(GazpreaParser::NotExprContext *ctx) {
        // std::cout << "visiting NotExpr in ASTBuilder";
        std::shared_ptr<ExprNode> expr = nullptr;
        std::string op;
        if (ctx->NOT()) {
            op = ctx->NOT()->getText();
        }
        if (ctx->expr()) {
            auto anyExpr = visit(ctx->expr());
            if (anyExpr.has_value()) {
            expr = safe_any_cast_ptr<ExprNode>(anyExpr);
            }
        }
        auto node = std::make_shared<NotExpr>(op, expr);
        setLocationFromCtx(node, ctx);
        node->constant.reset();
        if (expr && expr->constant) {
            auto res = gazprea::computeUnaryNumeric(*expr->constant, op);
            if (res) {
            node->constant = *res;
            node->type = node->constant->type;
            }
        }
        return expr_any(std::move(node));
        }
    std::any ASTBuilder::visitCompExpr(GazpreaParser::CompExprContext *ctx) {
        std::shared_ptr<ExprNode> left = nullptr;
        std::shared_ptr<ExprNode> right = nullptr;

        if (ctx->expr().size() >= 1) {
            auto anyLeft = visit(ctx->expr(0));
            if (anyLeft.has_value()) {
            left = safe_any_cast_ptr<ExprNode>(anyLeft);
            }
        }
        if (ctx->expr().size() >= 2) {
            auto anyRight = visit(ctx->expr(1));
            if (anyRight.has_value()) {
            right = safe_any_cast_ptr<ExprNode>(anyRight);
            }
        }
        std::string opText;
        if (ctx->LT()) {
            opText = ctx->LT()->getText();
        } else if (ctx->LTE()) {
            opText = ctx->LTE()->getText();
        } else if (ctx->GT()) {
            opText = ctx->GT()->getText();
        } else if (ctx->GTE()) {
            opText = ctx->GTE()->getText();
        }

        auto node = std::make_shared<CompExpr>(opText, left, right);
        setLocationFromCtx(node, ctx);
        node->constant.reset();
        if (left && left->constant && right && right->constant) {
            auto res =
                gazprea::computeBinaryComp(*left->constant, *right->constant, opText);
            if (res) {
            node->constant = *res;
            node->type = node->constant->type;
            }
        }
        return expr_any(std::move(node));
        }
    std::any ASTBuilder::visitAddExpr(GazpreaParser::AddExprContext *ctx) {
        std::shared_ptr<ExprNode> left = nullptr;
        std::shared_ptr<ExprNode> right = nullptr;

        if (ctx->expr().size() >= 1) {
            auto anyLeft = visit(ctx->expr(0));
            if (anyLeft.has_value()) {
            left = safe_any_cast_ptr<ExprNode>(anyLeft);
            }
        }
        if (ctx->expr().size() >= 2) {
            auto anyRight = visit(ctx->expr(1));
            if (anyRight.has_value()) {
            right = safe_any_cast_ptr<ExprNode>(anyRight);
            }
        }

        std::string opText;
        if (ctx->ADD()) {
            opText = ctx->ADD()->getText();
        } else if (ctx->MINUS()) {
            opText = ctx->MINUS()->getText();
        }

        auto node = std::make_shared<AddExpr>(opText, left, right);
        setLocationFromCtx(node, ctx);
        node->constant.reset();
        if (left && left->constant && right && right->constant) {
            auto res = gazprea::computeBinaryNumeric(*left->constant, *right->constant,
                                                    opText);
            if (res) {
            node->constant = *res;
            node->type = node->constant->type;
            }
        }
        return expr_any(std::move(node));
        }
    std::any ASTBuilder::visitMultExpr(GazpreaParser::MultExprContext *ctx) {
        std::shared_ptr<ExprNode> left = nullptr;
        std::shared_ptr<ExprNode> right = nullptr;

        if (ctx->expr().size() >= 1) {
            auto anyLeft = visit(ctx->expr(0));
            if (anyLeft.has_value()) {
            left = safe_any_cast_ptr<ExprNode>(anyLeft);
            }
        }
        if (ctx->expr().size() >= 2) {
            auto anyRight = visit(ctx->expr(1));
            if (anyRight.has_value()) {
            right = safe_any_cast_ptr<ExprNode>(anyRight);
            }
        }
        std::string opText;
        if (ctx->MULT()) {
            opText = ctx->MULT()->getText();
        } else if (ctx->DIV()) {
            opText = ctx->DIV()->getText();
        } else if (ctx->REM()) {
            opText = ctx->REM()->getText();
        } else if (ctx->op) {
            opText = ctx->op->getText();
        }
        auto node = std::make_shared<MultExpr>(opText, left, right);
        setLocationFromCtx(node, ctx);

        node->constant.reset();
        if (left && left->constant && right && right->constant) {
            auto res = gazprea::computeBinaryNumeric(*left->constant, *right->constant,
                                                    opText);
            if (res) {
            node->constant = *res;
            node->type = node->constant->type;
            }
        }
        return expr_any(std::move(node));
        }
    std::any ASTBuilder::visitAndExpr(GazpreaParser::AndExprContext *ctx) {
        std::shared_ptr<ExprNode> left = nullptr;
        std::shared_ptr<ExprNode> right = nullptr;

        if (ctx->expr().size() >= 1) {
            auto anyLeft = visit(ctx->expr(0));
            if (anyLeft.has_value()) {
            left = safe_any_cast_ptr<ExprNode>(anyLeft);
            }
        }
        if (ctx->expr().size() >= 2) {
            auto anyRight = visit(ctx->expr(1));
            if (anyRight.has_value()) {
            right = safe_any_cast_ptr<ExprNode>(anyRight);
            }
        }
        std::string opText;
        if (ctx->AND()) {
            opText = ctx->AND()->getText();
        }
        auto node = std::make_shared<AndExpr>(opText, left, right);
        setLocationFromCtx(node, ctx);
        node->constant.reset();
        if (left && left->constant && right && right->constant) {
            auto res =
                gazprea::computeBinaryComp(*left->constant, *right->constant, opText);
            if (res) {
            node->constant = *res;
            node->type = node->constant->type;
            }
        }
        return expr_any(std::move(node));
        }
    std::any ASTBuilder::visitOrExpr(GazpreaParser::OrExprContext *ctx) {
        std::shared_ptr<ExprNode> left = nullptr;
        std::shared_ptr<ExprNode> right = nullptr;

        if (ctx->expr().size() >= 1) {
            auto anyLeft = visit(ctx->expr(0));
            if (anyLeft.has_value()) {
            left = safe_any_cast_ptr<ExprNode>(anyLeft);
            }
        }
        if (ctx->expr().size() >= 2) {
            auto anyRight = visit(ctx->expr(1));
            if (anyRight.has_value()) {
            right = safe_any_cast_ptr<ExprNode>(anyRight);
            }
        }
        std::string opText;
        if (ctx->OR()) {
            opText = ctx->OR()->getText();
        } else if (ctx->XOR()) {
            opText = ctx->XOR()->getText();
        }
        auto node = std::make_shared<OrExpr>(opText, left, right);
        setLocationFromCtx(node, ctx);
        // Clear any previous constant annotation
        node->constant.reset();
        // If both children were annotated as constants, try to compute a folded value
        if (left && left->constant && right && right->constant) {
            auto res =
                gazprea::computeBinaryComp(*left->constant, *right->constant, opText);
            if (res) {
            node->constant = *res;
            node->type = node->constant->type;
            }
        }
        return expr_any(std::move(node));
        }
    std::any ASTBuilder::visitDotExpr(GazpreaParser::DotExprContext *ctx) {
        std::shared_ptr<ExprNode> left = nullptr;
        std::shared_ptr<ExprNode> right = nullptr;

        if (ctx->expr().size() >= 1) {
            auto anyLeft = visit(ctx->expr(0));
            if (anyLeft.has_value()) left = safe_any_cast_ptr<ExprNode>(anyLeft);
        }
        if (ctx->expr().size() >= 2) {
            auto anyRight = visit(ctx->expr(1));
            if (anyRight.has_value()) right = safe_any_cast_ptr<ExprNode>(anyRight);
        }

        std::string opText;
        if (ctx->DOTPROD()) opText = ctx->DOTPROD()->getText();

        auto node = std::make_shared<DotExpr>(opText, left, right);
        setLocationFromCtx(node, ctx);
        node->constant.reset();
        // dot product constant folding is performed in the ConstantFolding pass
        return expr_any(std::move(node));
    }    std::any ASTBuilder::visitConcatExpr(GazpreaParser::ConcatExprContext *ctx) {
        std::shared_ptr<ExprNode> left = nullptr;
        std::shared_ptr<ExprNode> right = nullptr;

        if (ctx->expr().size() >= 1) {
            auto anyLeft = visit(ctx->expr(0));
            if (anyLeft.has_value()) {
            left = safe_any_cast_ptr<ExprNode>(anyLeft);
            }
        }
        if (ctx->expr().size() >= 2) {
            auto anyRight = visit(ctx->expr(1));
            if (anyRight.has_value()) {
            right = safe_any_cast_ptr<ExprNode>(anyRight);
            }
        }
        std::string opText;
        if (ctx->CONCAT()) {
            opText = ctx->CONCAT()->getText();
        }
        auto node = std::make_shared<ConcatExpr>(opText, left, right);
        setLocationFromCtx(node, ctx);
        // Constant folding for string concatenation can be implemented here if needed
        // For now, reset constant.
        node->constant.reset();
        
        return expr_any(std::move(node));
    }

}