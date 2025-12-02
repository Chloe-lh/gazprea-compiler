#include "AST.h"
#include "ASTBuilderUtils.h"
#include "CompileTimeExceptions.h"
#include "GazpreaParser.h"
#include "ASTBuilder.h"
#include "Types.h"
#include <any>
#include <memory>
#include <ostream>
#include <stdexcept>

namespace gazprea{
  std::any ASTBuilder::visitArrayAccessAssignStat(GazpreaParser::ArrayAccessAssignStatContext *ctx){
    if(!ctx || !ctx->array_access() || !ctx->expr()) throw std::runtime_error("visitArrayAccessAssignStat: invalid context");
    GazpreaParser::Array_accessContext *ac = ctx->array_access();
    std::string id;
    int index;
    if(ac){
      if(ac->ID()) id = ac->ID()->getText();
      if(ac->INT()){
        try{
          index = std::stoi(ac->INT()->getText());
        }catch(const std::exception &){
          index = 0;
        }
      }
    }
    
    auto lhs = std::make_shared<ArrayAccessExpr>(id, index);
    auto rhsAny = visit(ctx->expr());
    auto rhs = safe_any_cast_ptr<ExprNode>(rhsAny);
    auto node = std::make_shared<ArrayAccessAssignStatNode>(std::move(lhs), std::move(rhs));
    setLocationFromCtx(node, ctx);
    return stat_any(node);
  }
  std::any ASTBuilder::visitArrayLitExpr(GazpreaParser::ArrayLitExprContext *ctx) {
    return visit(ctx->array_literal());
  }
  std::any ASTBuilder::visitArrayAccessExpr(GazpreaParser::ArrayAccessExprContext *ctx){
    std::string id="";
    int index;
    if(ctx->ID()){
      id =ctx->ID()->getText();
    }
    if (ctx->INT()) {
      try{
        index = std::stoi(ctx->INT()->getText());
      }catch(const std::exception &){
        index = 0;
      }
    }
    auto node = std::make_shared<ArrayAccessExpr>(id, index);
    node->type = CompleteType(BaseType::ARRAY);
    setLocationFromCtx(node, ctx);
    return expr_any(std::move(node));
  }

  std::any ASTBuilder::visitArrayStrideExpr(GazpreaParser::ArrayStrideExprContext *ctx){
    std::string id="";
    std::shared_ptr<ExprNode> expr = nullptr;
    if(ctx->ID()) id = ctx->ID()->getText();
    if (ctx->expr()) {
      auto anyExpr = visit(ctx->expr());
      if(anyExpr.has_value()) expr = safe_any_cast_ptr<ExprNode>(anyExpr);
    }
    std::shared_ptr<ArrayStrideExpr> node = std::make_shared<ArrayStrideExpr>(id, expr);
    node->type = CompleteType(BaseType::ARRAY);
    setLocationFromCtx(node, ctx);
    return expr_any(std::move(node));
  };
  std::any ASTBuilder::visitArraySliceExpr(GazpreaParser::ArraySliceExprContext *ctx){
    std::string id="";
    std::shared_ptr<RangeExprNode> range = nullptr;
    if(ctx->ID()) id = ctx->ID()->getText();
    if(ctx->rangeExpr()){
      auto anyRange = visit(ctx->rangeExpr());
      if(anyRange.has_value()){
        range = safe_any_cast_ptr<RangeExprNode>(anyRange);
      }
    }
    std::shared_ptr<ArraySliceExpr> node = std::make_shared<ArraySliceExpr>(id, range);
    node->type = CompleteType(BaseType::ARRAY);
    setLocationFromCtx(node, ctx);
    return expr_any(std::move(node));
  };

  std::any ASTBuilder::visitArray_literal(GazpreaParser::Array_literalContext *ctx){
    std::shared_ptr<ExprListNode> list = nullptr; //optional expression list
    if (ctx->exprList()) {
      auto anyList = visit(ctx->exprList());
      if (anyList.has_value()) {
        list = safe_any_cast_ptr<ExprListNode>(anyList);
      }
    }
    auto node = std::make_shared<ArrayLiteralNode>(list);
    node->type = CompleteType(BaseType::ARRAY);
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