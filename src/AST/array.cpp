#include "AST.h"
#include "ASTBuilderUtils.h"
#include "GazpreaParser.h"
#include "ASTBuilder.h"
#include "Types.h"
#include <any>
#include <memory>

namespace gazprea{
  
  std::any ASTBuilder::visitArrayAccessExpr(GazpreaParser::ArrayAccessExprContext *ctx){
    std::string id="";
    std::shared_ptr<ExprNode> expr = nullptr;
    if(ctx->ID()){
      id =ctx->ID()->getText();
    }
    if (ctx->expr()) {
      auto anyExpr = visit(ctx->expr());
      if(anyExpr.has_value()) expr = safe_any_cast_ptr<ExprNode>(anyExpr);
    }
    auto node = std::make_shared<ArrayAccessExpr>(id, expr);
    return expr_any(node);
  }
  std::any ASTBuilder::visitArrayTypedDec(GazpreaParser::ArrayTypedDecContext *ctx){
    std::string qual;
    std::shared_ptr<ArrayTypeNode> typeNode = nullptr;
    std::string id = "";
    if(ctx->qualifier()){
      qual = ctx->qualifier()->getText();
    }else{
      qual = "const";
    }
    if(ctx->ID()){
      id = ctx->ID()->getText();
    }
    if(ctx->array_type()){
      auto anyType = visit(ctx->array_type());
      if (anyType.has_value()) {
        typeNode = safe_any_cast_ptr<ArrayTypeNode>(anyType);
      }
    }
    std::shared_ptr<ArrayInitNode> init = nullptr;
    if (ctx->array_init()) {
      auto anyInit = visit(ctx->array_init());
      if (anyInit.has_value()) {
        init = safe_any_cast_ptr<ArrayInitNode>(anyInit);
      }
    }
    std::shared_ptr<ArrayTypedDecNode> node;
    if (init) {
      node = std::make_shared<ArrayTypedDecNode>(qual, id, typeNode, init);
    } else {
      node = std::make_shared<ArrayTypedDecNode>(qual, id, typeNode);
    }
    node->resolvedType = CompleteType(BaseType::ARRAY);
    // node->type = CompleteType(BaseType::ARRAY);
    setLocationFromCtx(node, ctx);
    return dec_any(std::move(node));
  }
  std::any ASTBuilder::visitArrayType(gazprea::GazpreaParser::Array_typeContext *ctx) {
    // get the element type
    auto elemAny = visit(ctx->type());
    auto elemType = safe_any_cast_ptr<TypeAliasNode>(elemAny);

    // extract size
    std::shared_ptr<ExprNode> sizeExpr = nullptr;
    bool isOpen = false;

    if (ctx->INT()) {
        // fixed size array
        int sizeValue = std::stoi(ctx->INT()->getText());
        sizeExpr = std::make_shared<IntNode>(sizeValue);

    } else if (ctx->MULT()) {
        // open array: '*'
        isOpen = true;
        // sizeExpr remains null
    }
    auto node = std::make_shared<ArrayTypeNode>(elemType, sizeExpr, isOpen);
    node->type = CompleteType(BaseType::ARRAY);

    setLocationFromCtx(node, ctx);
    return node;
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
        range = safe_any_cast_ptr<RangeExprNode>(range);
      }
    }
    std::shared_ptr<ArraySliceExpr> node = std::make_shared<ArraySliceExpr>(id, range);
    node->type = CompleteType(BaseType::ARRAY);
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