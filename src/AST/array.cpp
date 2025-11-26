#include "AST.h"
#include "ASTBuilderUtils.h"
#include "GazpreaParser.h"
#include "ASTBuilder.h"
#include <any>
#include <memory>

namespace gazprea{
  std::any ASTBuilder::visitArrayAccessExpr(GazpreaParser::ArrayAccessExprContext *ctx){
    std::string id = nullptr;
    std::shared_ptr<ExprNode> expr = nullptr;
    if(ctx->ID()){
      id =ctx->ID()->getText();
    }
    if(ctx->expr()){
      auto exprAny = visit(ctx->expr());
      if(exprAny.has_value()){
        expr = safe_any_cast_ptr<ExprNode>(exprAny);
      }
    }
    auto node = std::make_shared<ArrayAccessExpr>(id, expr);
    setLocationFromCtx(node, ctx);
    return expr_any(std::move(node));
  }
  std::any ASTBuilder::visitArrayInit(GazpreaParser::Array_initContext *ctx) {
    if (!ctx) return std::any();
    // Case: identifier initializer (e.g. = someArray)
    if (ctx->ID()) {
      std::string id = ctx->ID()->getText();
      auto node = std::make_shared<ArrayInitNode>(id);
      setLocationFromCtx(node, ctx);
      return node_any(std::move(node)); // ArrayInitNode is an ASTNode
    }
    // Case: array literal initializer (e.g. = [1,2,3])
    if (ctx->array_literal()) {
      auto anyLit = visit(ctx->array_literal());
      auto lit = safe_any_cast_ptr<ArrayLiteralNode>(anyLit);
      auto node = std::make_shared<ArrayInitNode>(lit);
      setLocationFromCtx(node, ctx);
      return node_any(std::move(node));
    }
    return std::any();
  }
  std::any ASTBuilder::visitArrayDec(GazpreaParser::Array_decContext *ctx){
    std::shared_ptr<ArrayTypeNode> type = nullptr;
    std::string id = "";
    std::shared_ptr<ArrayInitNode> init = nullptr;
    if(ctx->ID()){
      id = ctx->ID()->getText();
    }
    if(ctx->array_type()){
      auto anyType = visit(ctx->array_type());
      if (anyType.has_value()) {
        type = safe_any_cast_ptr<ArrayTypeNode>(anyType);
      }
    }
    if (ctx->array_init()) {
      auto anyInit = visit(ctx->array_init());
      if (anyInit.has_value()) {
        init = safe_any_cast_ptr<ArrayInitNode>(anyInit);
      }
    }
    std::shared_ptr<ArrayDecNode> node;
    if (init) {
      node = std::make_shared<ArrayDecNode>(id, type, init);
    } else {
      node = std::make_shared<ArrayDecNode>(id, type);
    }
    setLocationFromCtx(node, ctx);
    return dec_any(std::move(node));
  }
  std::any ASTBuilder::visitArrayType(GazpreaParser::Array_typeContext *ctx){
    return std::string("not yet implemented");
  }
  std::any ASTBuilder::visitArrayStrideExpr(GazpreaParser::ArrayStrideExprContext *ctx){
    std::string id = nullptr;
    std::shared_ptr<ExprNode> expr = nullptr;
    if(ctx->ID()) id = ctx->ID()->getText();
    // if(ctx->expr()){

    // }
     return std::string("not yet implemented");
    
  };
  std::any ASTBuilder::visitArraySliceExpr(GazpreaParser::ArraySliceExprContext *ctx){
    std::string id = nullptr;
    std::shared_ptr<RangeExprNode> range = nullptr;
    if(ctx->ID()) id = ctx->ID()->getText();
    if(ctx->rangeExpr()){
      auto anyRange = visit(ctx->rangeExpr());
      if(anyRange.has_value()){
        range = safe_any_cast_ptr<RangeExprNode>(range);
      }
    }
    std::shared_ptr<ArraySliceExpr> node = std::make_shared<ArraySliceExpr>(id, range);
    setLocationFromCtx(node, ctx);
    return expr_any(std::move(node));
  };

  std::any ASTBuilder::visitArrayLiteral(GazpreaParser::Array_literalContext *ctx){
    std::shared_ptr<ExprListNode> list = nullptr; //optional expression list
    if (ctx->exprList()) {
      auto anyList = visit(ctx->exprList());
      if (anyList.has_value()) {
        list = safe_any_cast_ptr<ExprListNode>(anyList);
      }
    }
    auto node = std::make_shared<ArrayLiteralNode>(list);
    setLocationFromCtx(node, ctx);
    return expr_any(std::move(node));
  };

  // exprList: expr (',' expr)*
  std::any ASTBuilder::visitExprList(GazpreaParser::ExprListContext *ctx) {
    std::vector<std::shared_ptr<ExprNode>> elements;
    for (auto exprCtx : ctx->expr()) {
      auto anyExpr = visit(exprCtx);
      auto expr = safe_any_cast_ptr<ExprNode>(anyExpr);
      elements.push_back(expr);
    }
    auto node = std::make_shared<ExprListNode>(std::move(elements));
    setLocationFromCtx(node, ctx);
    return node_any(std::move(node));
  };

  std::any ASTBuilder::visitRangeExpr(gazprea::GazpreaParser::RangeExprContext *ctx){
    std::shared_ptr<ExprNode> start = nullptr;
    std::shared_ptr<ExprNode> end = nullptr;
    auto startAny = visit(ctx->expr(0));
    auto endAny = visit(ctx->expr(1));
    if(startAny.has_value()){
      try {
        start = safe_any_cast_ptr<ExprNode>(startAny);
      } catch (const std::bad_any_cast&) {
        start = nullptr;
      }
    }
    if(endAny.has_value()){
      end = safe_any_cast_ptr<ExprNode>(endAny);
    }
    auto node = std::make_shared<RangeExprNode>(start, end);
    setLocationFromCtx(node, ctx);
    return expr_any(std::move(node));
  };
}