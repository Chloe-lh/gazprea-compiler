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
    std::shared_ptr<ExprNode> indexExpr = nullptr;
    std::shared_ptr<ExprNode> indexExpr2 = nullptr;
    if(ac){
      if(ac->ID()) id = ac->ID()->getText();
      auto exprs = ac->expr();
      if(!exprs.empty()){
        auto indexAny = visit(exprs[0]);
        indexExpr = safe_any_cast_ptr<ExprNode>(indexAny);
        
        if(exprs.size() >= 2){
          auto indexAny2 = visit(exprs[1]);
          indexExpr2 = safe_any_cast_ptr<ExprNode>(indexAny2);
        }
      }
    }
    
    std::shared_ptr<ArrayAccessNode> lhs;
    if(indexExpr2){
      lhs = std::make_shared<ArrayAccessNode>(id, indexExpr, indexExpr2);
    } else {
      lhs = std::make_shared<ArrayAccessNode>(id, indexExpr);
    }
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
    std::shared_ptr<ExprNode> indexExpr = nullptr;
    std::shared_ptr<ExprNode> indexExpr2 = nullptr;
    auto aa = ctx->array_access();
    if(aa->ID()){
      id=aa->ID()->getText();
    }
    // Get all index expressions - there may be two for matrices
    auto exprs = aa->expr();
    if(!exprs.empty()){
      auto indexAny = visit(exprs[0]);
      indexExpr = safe_any_cast_ptr<ExprNode>(indexAny);
      
      if(exprs.size() >= 2){
        auto indexAny2 = visit(exprs[1]);
        indexExpr2 = safe_any_cast_ptr<ExprNode>(indexAny2);
      }
    }
    
    std::shared_ptr<ArrayAccessNode> node;
    if(indexExpr2){
      node = std::make_shared<ArrayAccessNode>(id, indexExpr, indexExpr2);
      node->type = CompleteType(BaseType::MATRIX);
    } else {
      node = std::make_shared<ArrayAccessNode>(id, indexExpr);
      node->type = CompleteType(BaseType::ARRAY);
    }
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
  // no need to check array literal for matrix
  std::any ASTBuilder::visitArray_literal(GazpreaParser::Array_literalContext *ctx){
    std::shared_ptr<ExprListNode> list = nullptr; //optional expression list
    if (ctx->exprList()) {
      auto anyList = visit(ctx->exprList());
      if (anyList.has_value()) {
        list = safe_any_cast_ptr<ExprListNode>(anyList);
      }
    }
    auto node = std::make_shared<ArrayLiteralNode>(list);
    if (list && !list->list.empty()) {
        //TODO this is very brute
        auto firstExprType = list->list[0]->type;  // make sure each ExprNode has type set!
        node->type = CompleteType(BaseType::ARRAY);
        node->type.subTypes.push_back(firstExprType);
    } else {
        node->type = CompleteType(BaseType::ARRAY); // empty array literal
    }
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

  // Generators
  std::any ASTBuilder::visitGeneratorDomain(GazpreaParser::GeneratorDomainContext *ctx) {
    std::string iter = ctx->ID()->getText();
    std::shared_ptr<ExprNode> dom = nullptr;
    if (ctx->rangeExpr()) {
      auto anyDom = visit(ctx->rangeExpr());
      if (anyDom.has_value()) dom = safe_any_cast_ptr<ExprNode>(anyDom);
    } else if (ctx->array_literal()) {
      auto anyDom = visit(ctx->array_literal());
      if (anyDom.has_value()) dom = safe_any_cast_ptr<ExprNode>(anyDom);
    } else if (ctx->expr()) {
      auto anyDom = visit(ctx->expr());
      if (anyDom.has_value()) dom = safe_any_cast_ptr<ExprNode>(anyDom);
    }
    return std::make_pair(iter, dom);
  }

  std::any ASTBuilder::visitGeneratorDomains(GazpreaParser::GeneratorDomainsContext *ctx) {
    std::vector<std::pair<std::string, std::shared_ptr<ExprNode>>> domains;
    for (auto dctx : ctx->generatorDomain()) {
      auto anyDom = visit(dctx);
      if (anyDom.has_value()) {
        domains.push_back(std::any_cast<std::pair<std::string, std::shared_ptr<ExprNode>>>(anyDom));
      }
    }
    return domains;
  }

  std::any ASTBuilder::visitGeneratorBody(GazpreaParser::GeneratorBodyContext *ctx) {
    auto anyDomains = visit(ctx->generatorDomains());
    auto domains = std::any_cast<std::vector<std::pair<std::string, std::shared_ptr<ExprNode>>>>(anyDomains);
    std::shared_ptr<ExprNode> rhs = nullptr;
    if (ctx->expr()) {
      auto anyRhs = visit(ctx->expr());
      if (anyRhs.has_value()) rhs = safe_any_cast_ptr<ExprNode>(anyRhs);
    }
    auto node = std::make_shared<GeneratorExprNode>(std::move(domains), rhs);
    setLocationFromCtx(node, ctx);
    return expr_any(std::move(node));
  }

  std::any ASTBuilder::visitGeneratorExpr(GazpreaParser::GeneratorExprContext *ctx) {
    return visit(ctx->generatorBody());
  }

  std::any ASTBuilder::visitRangeExpr(gazprea::GazpreaParser::RangeExprContext *ctx){
    std::shared_ptr<ExprNode> start = nullptr;
    std::shared_ptr<ExprNode> end = nullptr;
    std::shared_ptr<ExprNode> step = nullptr;

    auto exprs = ctx->expr();

    // If BY is present, the last expr is the stride; remove it from start/end processing
    if (ctx->BY() && !exprs.empty()) {
      auto anyStep = visit(exprs.back());
      if (anyStep.has_value()) {
        step = safe_any_cast_ptr<ExprNode>(anyStep);
      }
      exprs.pop_back();
    }

    if (!exprs.empty()) {
      if (exprs.size() == 1) {
        // Either RANGE expr (..end) or expr RANGE (start..)
        if (ctx->getStart()->getType() == gazprea::GazpreaParser::RANGE) {
          auto endAny = visit(exprs[0]);
          if (endAny.has_value()) {
            try { end = safe_any_cast_ptr<ExprNode>(endAny); } catch (...) { end = nullptr; }
          }
        } else {
          auto startAny = visit(exprs[0]);
          if (startAny.has_value()) {
            try { start = safe_any_cast_ptr<ExprNode>(startAny); } catch (...) { start = nullptr; }
          }
        }
      } else {
        // Two or more exprs: use first as start, second as end
        auto startAny = visit(exprs[0]);
        auto endAny = visit(exprs[1]);
        if (startAny.has_value()) {
          try { start = safe_any_cast_ptr<ExprNode>(startAny); } catch (...) { start = nullptr; }
        }
        if (endAny.has_value()) {
          try { end = safe_any_cast_ptr<ExprNode>(endAny); } catch (...) { end = nullptr; }
        }
      }
    }

    auto node = std::make_shared<RangeExprNode>(start, end, step);
    setLocationFromCtx(node, ctx);
    return expr_any(std::move(node));
  };
}