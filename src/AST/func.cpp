#include "ASTBuilder.h"
#include "AST.h"
#include "ASTBuilderUtils.h"
#include "GazpreaParser.h"
#include "Scope.h"
#include "Types.h"
#include <any>
#include <memory>
#include <stdlib.h>

namespace gazprea{
//  | CALL ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT END  #CallStat
std::any ASTBuilder::visitFuncCallExpr(GazpreaParser::FuncCallExprContext *ctx) {
  std::string funcName = ctx->ID()->getText();
  auto args = gazprea::collectArgs(*this, ctx->expr());
  auto node = std::make_shared<FuncCallExpr>(funcName, std::move(args));
  setLocationFromCtx(node, ctx);
  return expr_any(std::move(node));
}

// since there is no function block node, return a Function Prototype with a
// body combines a function signature with a function body
std::any
ASTBuilder::visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  // Extract params and convert to VarInfo vector (parameters are const by
  // default)
  std::vector<std::pair<CompleteType, std::string>> params =
      ExtractParams(*this, ctx);
  // Convert parser-style param list to VarInfo vector expected by the AST
  // FuncNode
  std::vector<VarInfo> varParams =
      ParamsToVarInfo(params, /*isConstDefault=*/true);
  std::shared_ptr<BlockNode> body = nullptr;
  CompleteType returnType = ExtractReturnType(*this, ctx);

  if (ctx->block()) { // function has a block body
    auto anyBody = visit(ctx->block());
    if (anyBody.has_value()) {
      body = safe_any_cast_ptr<BlockNode>(anyBody);
    }
  }

  // Create a FuncBlockNode (function with a block body)
  auto node =
      std::make_shared<FuncBlockNode>(funcName, varParams, returnType, body);
  return node_any(std::move(node));
}

// combines a function signature with a function body
std::any ASTBuilder::visitFunctionBlockTupleReturn(
    GazpreaParser::FunctionBlockTupleReturnContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  std::vector<std::pair<CompleteType, std::string>> params =
      ExtractParams(*this, ctx);
  // Convert to VarInfo expected by AST FuncNode constructors
  std::vector<VarInfo> varParams =
      ParamsToVarInfo(params, /*isConstDefault=*/true);
  std::shared_ptr<BlockNode> body = nullptr;
  CompleteType returnType =
      gazprea::ExtractReturnType(*this, ctx);

  if (ctx->block()) { // function has a block
    auto anyBody = visit(ctx->block());
    if (anyBody.has_value()) {
      body = safe_any_cast_ptr<BlockNode>(anyBody);
    }
  }
  auto node =
      std::make_shared<FuncBlockNode>(funcName, varParams, returnType, body);
  return node_any(std::move(node));
}

std::any 
ASTBuilder::visitFunctionBlockStructReturn(GazpreaParser::FunctionBlockStructReturnContext *ctx) {
      throw std::runtime_error("visitFunctionBlockStructReturn: not implemented");
}

std::any ASTBuilder::visitProcedurePrototype(
    GazpreaParser::ProcedurePrototypeContext *ctx) {
  std::string procName = ctx->ID()->getText();
  auto tuples =
      gazprea::ExtractParamsWithQualifiers(*this, ctx->param());
  std::vector<VarInfo> varParams;
  varParams.reserve(tuples.size());
  for (size_t i = 0; i < tuples.size(); ++i) {
    CompleteType ptype(BaseType::UNKNOWN);
    std::string pname;
    bool isConst = true;
    std::tie(ptype, pname, isConst) = tuples[i];
    if (pname.empty()) {
      pname = "_arg" + std::to_string(i);
    }
    varParams.emplace_back(pname, ptype, isConst);
  }

  CompleteType returnType(BaseType::UNKNOWN);
  if (ctx->RETURNS() && ctx->type()) {
    auto anyRet = visit(ctx->type());
    if (anyRet.has_value() && anyRet.type() == typeid(CompleteType)) {
      try {
        returnType = std::any_cast<CompleteType>(anyRet);
      } catch (const std::bad_any_cast &) {
        returnType = CompleteType(BaseType::UNKNOWN);
      }
    }
  }
  auto node =
      std::make_shared<ProcedurePrototypeNode>(procName, varParams, returnType);
      setLocationFromCtx(node, ctx);
  // no body for a prototype
  return node_any(std::move(node));
}
// PROCEDURE ID PARENLEFT (param (COMMA param)*)? PARENRIGHT (RETURNS type)?
// block;
std::any
ASTBuilder::visitProcedureBlock(GazpreaParser::ProcedureBlockContext *ctx) {
  std::string funcName = ctx->ID()->getText();

  // Extract params with qualifiers via shared helper
  std::vector<VarInfo> varParams;
  auto tuples =
      gazprea::ExtractParamsWithQualifiers(*this, ctx->param());
  varParams.reserve(tuples.size());
  for (size_t i = 0; i < tuples.size(); ++i) {
    CompleteType ptype(BaseType::UNKNOWN);
    std::string pname;
    bool isConst = true;
    std::tie(ptype, pname, isConst) = tuples[i];
    if (pname.empty()) {
      pname = "_arg" + std::to_string(i);
    }
    varParams.emplace_back(pname, ptype, isConst);
  }

  std::shared_ptr<BlockNode> body = nullptr;

  // Handle procedure optional return type
  CompleteType returnType = CompleteType(BaseType::UNKNOWN);
  if (ctx->RETURNS()) {
    if (ctx->type()) {
      auto anyRet = visit(ctx->type());
      if (anyRet.has_value() && anyRet.type() == typeid(CompleteType)) {
        try {
          returnType = std::any_cast<CompleteType>(anyRet);
        } catch (const std::bad_any_cast &) {
          returnType = CompleteType(BaseType::UNKNOWN);
        }
      }
    }
  }

  if (ctx->block()) { // procedure has a block body
    auto anyBody = visit(ctx->block());
    if (anyBody.has_value()) {
      body = safe_any_cast_ptr<BlockNode>(anyBody);
    }
  }

  auto node = std::make_shared<ProcedureBlockNode>(funcName, varParams,
                                                   returnType, body);
                                                   setLocationFromCtx(node, ctx);
  return node_any(std::move(node));
}

std::any ASTBuilder::visitFunctionPrototype(
    GazpreaParser::FunctionPrototypeContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  std::vector<std::pair<CompleteType, std::string>> params =
      gazprea::ExtractParams(*this, ctx);
  std::vector<VarInfo> varParams =
      gazprea::ParamsToVarInfo(params, /*isConstDefault=*/true);
  CompleteType returnType =
      gazprea::ExtractReturnType(*this, ctx);
  auto node =
      std::make_shared<FuncPrototypeNode>(funcName, varParams, returnType);
      setLocationFromCtx(node, ctx);
  // no body for a prototype
  return node_any(std::move(node));
}
std::any ASTBuilder::visitFunctionPrototypeTupleReturn(
    GazpreaParser::FunctionPrototypeTupleReturnContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  std::vector<std::pair<CompleteType, std::string>> params =
      gazprea::ExtractParams(*this, ctx);
  std::vector<VarInfo> varParams =
      gazprea::ParamsToVarInfo(params, /*isConstDefault=*/true);
  CompleteType returnType =
      gazprea::ExtractReturnType(*this, ctx);
  // no body for a prototype
  auto node =
      std::make_shared<FuncPrototypeNode>(funcName, varParams, returnType);
      setLocationFromCtx(node, ctx);
  return node_any(std::move(node));
}

std::any
ASTBuilder::visitFunctionPrototypeStructReturn(
    GazpreaParser::FunctionPrototypeStructReturnContext *ctx) {
  throw std::runtime_error("visitFunctionPrototypeStructReturn: not implemented");
}

std::any
ASTBuilder::visitFunctionStat(GazpreaParser::FunctionStatContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  std::vector<std::pair<CompleteType, std::string>> params =
      gazprea::ExtractParams(*this, ctx);
  std::vector<VarInfo> varParams =
      gazprea::ParamsToVarInfo(params, /*isConstDefault=*/true);
  CompleteType returnType =
      gazprea::ExtractReturnType(*this, ctx);
  std::shared_ptr<StatNode> returnStat = nullptr;
  if (ctx->expr()) {
    auto anyExpr = visit(ctx->expr());
    auto exprNode = safe_any_cast_ptr<ExprNode>(anyExpr);
    setLocationFromCtx(exprNode, ctx);
    if (exprNode) {
      auto retNode = std::make_shared<ReturnStatNode>(exprNode);
      setLocationFromCtx(exprNode, ctx);
      returnStat = retNode;
    }
  }
  // construct FuncStatNode using VarInfo vector
  auto node = std::make_shared<FuncStatNode>(funcName, varParams, returnType,
                                             returnStat);
                                             setLocationFromCtx(node, ctx);
  return node_any(std::move(node));
}
}