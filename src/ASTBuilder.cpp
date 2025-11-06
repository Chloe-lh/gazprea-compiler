#include "ASTBuilder.h"
#include "AST.h"
#include "ASTBuilderUtils.h"
#include "CompileTimeExceptions.h"
#include "ConstantHelpers.h"
#include "GazpreaParser.h"
#include "Scope.h"
#include "Types.h"
#include "antlr4-runtime.h"
#include <any>
#include <memory>
#include <stdexcept>
#include <stdlib.h>
#include <type_traits>

namespace gazprea {

// Helper to return an AST node wrapped in std::any with an upcast to the
// common base `ASTNode`. Use this when a visitor wants to return a concrete
// node but callers expect a `std::shared_ptr<ASTNode>` inside the any.
template <typename T> static inline std::any node_any(std::shared_ptr<T> n) {
  return std::static_pointer_cast<ASTNode>(std::move(n));
}
// Helpers that canonicalize the std::any payload for different AST families.
template <typename T> static inline std::any expr_any(std::shared_ptr<T> n) {
  return std::static_pointer_cast<ExprNode>(std::move(n));
}
template <typename T> static inline std::any stat_any(std::shared_ptr<T> n) {
  return std::static_pointer_cast<StatNode>(std::move(n));
}
template <typename T> static inline std::any dec_any(std::shared_ptr<T> n) {
  return std::static_pointer_cast<DecNode>(std::move(n));
}

// Small helper for safely extracting shared_ptr<T> from a std::any produced
// by the builder. Avoids repeating try/catch everywhere.
template <typename T> static inline std::shared_ptr<T>
safe_any_cast_ptr(const std::any &a) {
  try {
    if (a.has_value() && a.type() == typeid(std::shared_ptr<T>))
      return std::any_cast<std::shared_ptr<T>>(a);
  } catch (const std::bad_any_cast &) {
    // fall through
  }
  return nullptr;
}

std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
  std::vector<std::shared_ptr<ASTNode>> nodes;
  for (auto child : ctx->children) {
    auto anyNode = visit(child);
    if (anyNode.has_value()) {
      auto node = safe_any_cast_ptr<ASTNode>(anyNode);
      if (node)
        nodes.push_back(node);
    }
  }
  auto node = std::make_shared<FileNode>(std::move(nodes));
  return node_any(std::move(node));
}
std::any ASTBuilder::visitBlock(GazpreaParser::BlockContext *ctx) {
  std::vector<std::shared_ptr<DecNode>> decs;
  std::vector<std::shared_ptr<StatNode>> stats;
  for (auto decCtx : ctx->dec()) {
    auto decAny = visit(decCtx);
    auto dec = safe_any_cast_ptr<DecNode>(decAny);
    if (dec)
      decs.push_back(dec);
  }
  for (auto statCtx : ctx->stat()) {
    auto statAny = visit(statCtx);
    auto stat = safe_any_cast_ptr<StatNode>(statAny);
    if (stat)
      stats.push_back(stat);
  }
  auto node = std::make_shared<BlockNode>(std::move(decs), std::move(stats));
  return node_any(std::move(node));
}
//| AS '<' tuple_dec  '>' PARENLEFT expr PARENRIGHT   #TupleTypeCastExpr
std::any ASTBuilder::visitTupleTypeCastExpr(GazpreaParser::TupleTypeCastExprContext *ctx){
  CompleteType targetTupleType = CompleteType(BaseType::UNKNOWN);
  if (ctx->tuple_dec()) {
    auto anyType = visit(ctx->tuple_dec());
    if (anyType.has_value() && anyType.type() == typeid(CompleteType)) {
      try {
        targetTupleType = std::any_cast<CompleteType>(anyType);
      } catch (const std::bad_any_cast &) {
        targetTupleType = CompleteType(BaseType::UNKNOWN);
      }
    }
  }
    // Build the expression operand
  std::shared_ptr<ExprNode> expr = nullptr;
  if (ctx->expr()) {
    auto anyExpr = visit(ctx->expr());
    if (anyExpr.has_value()) {
      expr = safe_any_cast_ptr<ExprNode>(anyExpr);
      
    }
  }
  auto node = std::make_shared<TupleTypeCastNode>(targetTupleType, expr);
  return expr_any(std::move(node));
}
// AS '<' type '>' PARENLEFT expr PARENRIGHT         #TypeCastExpr
std::any ASTBuilder::visitTypeCastExpr(GazpreaParser::TypeCastExprContext *ctx) {
  // Determine the target type (returns a CompleteType from visitType)
  CompleteType target = CompleteType(BaseType::UNKNOWN);
  if (ctx->type()) {
    auto anyType = visit(ctx->type());
    if (anyType.has_value() && anyType.type() == typeid(CompleteType)) {
      try {
        target = std::any_cast<CompleteType>(anyType);
      } catch (const std::bad_any_cast &) {
        target = CompleteType(BaseType::UNKNOWN);
      }
    } else {
      // Fallback
      target = CompleteType(BaseType::UNKNOWN);
    }
  }

  // Build the expression operand
  std::shared_ptr<ExprNode> expr = nullptr;
  if (ctx->expr()) {
    auto anyExpr = visit(ctx->expr());
    if (anyExpr.has_value()) {
      expr = safe_any_cast_ptr<ExprNode>(anyExpr);
    }
  }
  auto node = std::make_shared<TypeCastNode>(target, expr);
  return expr_any(std::move(node));
}
/*
dec
    : qualifier? type ID (EQ expr)? END          #ExplicitTypedDec
    | qualifier ID EQ expr END                   #InferredTypeDec
    | qualifier? tuple_dec ID (EQ expr)? END     #TupleTypedDec
*/
std::any
ASTBuilder::visitExplicitTypedDec(GazpreaParser::ExplicitTypedDecContext *ctx) {
  // qualifier (optional)
  std::string qualifier;
  if (ctx->qualifier()) {
    auto qualAny = visit(ctx->qualifier());
    if (qualAny.has_value()) {
      try {
        qualifier = std::any_cast<std::string>(qualAny);
      } catch (const std::bad_any_cast &) {
        qualifier = "";
      }
    }
  }
  std::shared_ptr<TypeAliasNode> typeAlias = nullptr;
  if (ctx->type()) {
    auto tctx = ctx->type();
    if (tctx->INTEGER()) {
      typeAlias = std::make_shared<TypeAliasNode>(
          std::string(""), CompleteType(BaseType::INTEGER));
    } else if (tctx->REAL()) {
      typeAlias = std::make_shared<TypeAliasNode>(std::string(""),
                                                  CompleteType(BaseType::REAL));
    } else if (tctx->BOOLEAN()) {
      typeAlias = std::make_shared<TypeAliasNode>(std::string(""),
                                                  CompleteType(BaseType::BOOL));
    } else if (tctx->CHARACTER()) {
      typeAlias = std::make_shared<TypeAliasNode>(
          std::string(""), CompleteType(BaseType::CHARACTER));
    } else if (tctx->ID()) {
      // named alias; store the alias name and leave concrete type unknown
      typeAlias = std::make_shared<TypeAliasNode>(
          tctx->ID()->getText(), CompleteType(BaseType::UNKNOWN));
    } else {
      typeAlias = std::make_shared<TypeAliasNode>(
          std::string(""), CompleteType(BaseType::UNKNOWN));
    }
  } else {
    typeAlias = std::make_shared<TypeAliasNode>(
        std::string(""), CompleteType(BaseType::UNKNOWN));
  }
  // declared identifier
  std::string id = ctx->ID()->getText();
  // optional initializer
  std::shared_ptr<ExprNode> init = nullptr;
  if (ctx->expr()) {
    auto exprAny = visit(ctx->expr());
    if (exprAny.has_value()) {
      init = safe_any_cast_ptr<ExprNode>(exprAny);
    }
  }
  auto node = std::make_shared<TypedDecNode>(id, typeAlias, qualifier, init);
  return node_any(std::move(node));
}
std::any
ASTBuilder::visitInferredTypeDec(GazpreaParser::InferredTypeDecContext *ctx) {
  // qualifier is returned as a std::any holding a std::string
  std::string qualifier;
  if (ctx->qualifier()) {
    auto qualAny = visit(ctx->qualifier());
    if (qualAny.has_value()) {
      try {
        qualifier = std::any_cast<std::string>(qualAny);
      } catch (const std::bad_any_cast &) {
        qualifier = "";
      }
    }
  }
  std::string id = ctx->ID()->getText();
  // The initializer expression (if present) is under ctx->expr()
  std::shared_ptr<ExprNode> expr = nullptr;
  if (ctx->expr()) {
    auto exprAny = visit(ctx->expr());
    if (exprAny.has_value()) {
      expr = safe_any_cast_ptr<ExprNode>(exprAny);
    }
  }
  auto node = std::make_shared<InferredDecNode>(qualifier, id, expr);
  return node_any(std::move(node));
}
std::any
ASTBuilder::visitTupleTypedDec(GazpreaParser::TupleTypedDecContext *ctx) {
  std::string id = ctx->ID()->getText();
  CompleteType tupleType = CompleteType(BaseType::UNKNOWN);
  if (ctx->tuple_dec()) {
    auto anyType = visit(ctx->tuple_dec());
    if (anyType.has_value()) {
      try {
        tupleType = std::any_cast<CompleteType>(anyType);
      } catch (const std::bad_any_cast &) {
        tupleType = CompleteType(BaseType::UNKNOWN);
      }
    }
  }

  // optional initializer expression
  std::shared_ptr<ExprNode> init = nullptr;
  if (ctx->expr()) {
    auto anyInit = visit(ctx->expr());
    if (anyInit.has_value()) {
      init = safe_any_cast_ptr<ExprNode>(anyInit);
    }
  }

  auto node = std::make_shared<TupleTypedDecNode>(id, tupleType);
  node->init = init;
  return node_any(std::move(node));
}
std::any
ASTBuilder::visitTupleAccessExpr(GazpreaParser::TupleAccessExprContext *ctx) {
  // tuple_access: ID DECIM TUPLE_INT
  auto ta = ctx->tuple_access();
  std::string tupleName = "";
  int index = 0;
  if (ta) {
    if (ta->ID())
      tupleName = ta->ID()->getText();
    if (ta->TUPLE_INT()) {
      try {
        index = std::stoi(ta->TUPLE_INT()->getText());
      } catch (const std::exception &) {
        index = 0;
      }
    }
  }
  auto node = std::make_shared<TupleAccessNode>(tupleName, index);
  return node_any(std::move(node));
}
//  TupleTypeAliasNode(const std::string& aliasName, CompleteType tupleType);
// TYPEALIAS tuple_dec ID
std::any
ASTBuilder::visitTupleTypeAlias(GazpreaParser::TupleTypeAliasContext *ctx) {
  std::string alias = ctx->ID()->getText();
  CompleteType tupleType(BaseType::UNKNOWN);
  if (ctx->tuple_dec()) {
    auto anyType = visit(ctx->tuple_dec());
    if (anyType.has_value() && anyType.type() == typeid(CompleteType)) {
      tupleType = std::any_cast<CompleteType>(anyType);
    } else {
      // Fallback to an unknown tuple type
      tupleType = CompleteType(BaseType::TUPLE);
    }
  }

  auto node = std::make_shared<TupleTypeAliasNode>(alias, tupleType);
  return node_any(std::move(node));
}
std::any ASTBuilder::visitQualifier(GazpreaParser::QualifierContext *ctx) {
  if (ctx->VAR()) {
    return std::string("var");
  } else if (ctx->CONST()) {
    return std::string("const");
  } else {
    return std::string("");
  }
}
// returns CompleteType object based on grammar else returns unknown
std::any ASTBuilder::visitType(GazpreaParser::TypeContext *ctx) {
  if (ctx->BOOLEAN())
    return CompleteType(BaseType::BOOL);
  if (ctx->ID())
    return CompleteType(BaseType::STRING);
  if (ctx->INTEGER())
    return CompleteType(BaseType::INTEGER);
  if (ctx->REAL())
    return CompleteType(BaseType::REAL);
  if (ctx->CHARACTER())
    return CompleteType(BaseType::CHARACTER);
  return CompleteType(BaseType::UNKNOWN);
}
std::any
ASTBuilder::visitBasicTypeAlias(GazpreaParser::BasicTypeAliasContext *ctx) {
  // Grammar: TYPEALIAS ID ID  -> alias the type named by ID(0) as ID(1)
  std::string referenced = ctx->ID(0)->getText();
  std::string aliasName = ctx->ID(1)->getText();

  CompleteType aliasedType(BaseType::UNKNOWN);
  if (referenced == "integer")
    aliasedType = CompleteType(BaseType::INTEGER);
  else if (referenced == "real")
    aliasedType = CompleteType(BaseType::REAL);
  else if (referenced == "boolean")
    aliasedType = CompleteType(BaseType::BOOL);
  else if (referenced == "character")
    aliasedType = CompleteType(BaseType::CHARACTER);

  auto node = std::make_shared<TypeAliasDecNode>(aliasName, aliasedType);
  // Records the original referenced name so later passes can resolve it
  // if aliasedType was left as UNKNOWN.
  node->declTypeName = referenced;
  return node_any(std::move(node));
}
std::any ASTBuilder::visitAssignStat(GazpreaParser::AssignStatContext *ctx) {
  std::string name = ctx->ID()->getText();
  auto exprAny = visit(ctx->expr());
  auto expr = safe_any_cast_ptr<ExprNode>(exprAny);
  auto node = std::make_shared<AssignStatNode>(name, expr);
  return stat_any(std::move(node));
}
std::any ASTBuilder::visitBreakStat(GazpreaParser::BreakStatContext *ctx) {
  auto node = std::make_shared<BreakStatNode>();
  return stat_any(std::move(node));
}
std::any
ASTBuilder::visitContinueStat(GazpreaParser::ContinueStatContext *ctx) {
  auto node = std::make_shared<ContinueStatNode>();
  return stat_any(std::move(node));
}
std::any ASTBuilder::visitReturnStat(GazpreaParser::ReturnStatContext *ctx) {
  std::shared_ptr<ExprNode> expr = nullptr;
  if (ctx->expr()) {
    auto anyExpr = visit(ctx->expr());
    if (anyExpr.has_value()) {
      expr = safe_any_cast_ptr<ExprNode>(anyExpr);
    }
  }
  auto node = std::make_shared<ReturnStatNode>(expr);
  return stat_any(std::move(node));
}
//  | CALL ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT END  #CallStat
std::any
ASTBuilder::visitFuncCallExpr(GazpreaParser::FuncCallExprContext *ctx) {
  std::string funcName = ctx->ID()->getText();
  auto args = gazprea::builder_utils::collectArgs(
      *this, std::vector<GazpreaParser::ExprContext *>(ctx->expr().begin(),
                                                       ctx->expr().end()));
  auto node = std::make_shared<FuncCallExpr>(funcName, std::move(args));
  return expr_any(std::move(node));
}
//  CALL ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT END  #CallStat
std::any ASTBuilder::visitCallStat(GazpreaParser::CallStatContext *ctx) {
  std::string funcName = ctx->ID()->getText();
  auto args = gazprea::builder_utils::collectArgs(
      *this, std::vector<GazpreaParser::ExprContext *>(ctx->expr().begin(),
                                                       ctx->expr().end()));

  // Build an expression-level call node and wrap it in a CallStatNode
  auto callExpr = std::make_shared<FuncCallExpr>(funcName, std::move(args));
  auto node = std::make_shared<CallStatNode>(callExpr);
  return stat_any(std::move(node));
}

std::any ASTBuilder::visitInputStat(GazpreaParser::InputStatContext *ctx) {
  std::string id = ctx->ID()->getText();
  auto node = std::make_shared<InputStatNode>(id);
  return stat_any(std::move(node));
}
std::any ASTBuilder::visitOutputStat(GazpreaParser::OutputStatContext *ctx) {
  std::shared_ptr<ExprNode> expr = nullptr;
  if (ctx->expr()) {
    auto anyExpr = visit(ctx->expr());
    if (anyExpr.has_value()) {
      expr = safe_any_cast_ptr<ExprNode>(anyExpr);
    }
  }
  auto node = std::make_shared<OutputStatNode>(expr);
  return stat_any(std::move(node));
}
// since there is no function block node, return a Function Prototype with a
// body combines a function signature with a function body
std::any
ASTBuilder::visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  // Extract params and convert to VarInfo vector (parameters are const by
  // default)
  std::vector<std::pair<CompleteType, std::string>> params =
      builder_utils::ExtractParams(*this, ctx);
  // Convert parser-style param list to VarInfo vector expected by the AST
  // FuncNode
  std::vector<VarInfo> varParams =
      builder_utils::ParamsToVarInfo(params, /*isConstDefault=*/true);
  std::shared_ptr<BlockNode> body = nullptr;
  CompleteType returnType = builder_utils::ExtractReturnType(*this, ctx);

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
      builder_utils::ExtractParams(*this, ctx);
  // Convert to VarInfo expected by AST FuncNode constructors
  std::vector<VarInfo> varParams =
      builder_utils::ParamsToVarInfo(params, /*isConstDefault=*/true);
  std::shared_ptr<BlockNode> body = nullptr;
  CompleteType returnType =
      gazprea::builder_utils::ExtractReturnType(*this, ctx);

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
// PROCEDURE ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT block;
std::any ASTBuilder::visitProcedure(GazpreaParser::ProcedureContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  // Extract params and convert to VarInfo vector (parameters are const by
  // default)
  std::vector<std::pair<CompleteType, std::string>> params =
      builder_utils::ExtractParams(*this, ctx);
  std::vector<VarInfo> varParams =
      builder_utils::ParamsToVarInfo(params, /*isConstDefault=*/true);
  std::shared_ptr<BlockNode> body = nullptr;

  // Procedures ordinarily have no return value; use UNKNOWN as a placeholder
  // return type here. If the grammar later supports annotated procedure
  // returns, update builder_utils and ASTBuilder accordingly.
  CompleteType returnType = CompleteType(BaseType::UNKNOWN);

  if (ctx->block()) { // procedure has a block body
    auto anyBody = visit(ctx->block());
    if (anyBody.has_value()) {
      body = safe_any_cast_ptr<BlockNode>(anyBody);
    }
  }

  auto node = std::make_shared<ProcedureNode>(funcName, varParams, returnType,
                                              body);
  return node_any(std::move(node));
}
std::any ASTBuilder::visitFunctionPrototype(
    GazpreaParser::FunctionPrototypeContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  std::vector<std::pair<CompleteType, std::string>> params =
      gazprea::builder_utils::ExtractParams(*this, ctx);
  std::vector<VarInfo> varParams =
      gazprea::builder_utils::ParamsToVarInfo(params, /*isConstDefault=*/true);
  CompleteType returnType =
      gazprea::builder_utils::ExtractReturnType(*this, ctx);
  auto node =
      std::make_shared<FuncPrototypeNode>(funcName, varParams, returnType);
  // no body for a prototype
  return node_any(std::move(node));
}
std::any ASTBuilder::visitFunctionPrototypeTupleReturn(
    GazpreaParser::FunctionPrototypeTupleReturnContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  std::vector<std::pair<CompleteType, std::string>> params =
      gazprea::builder_utils::ExtractParams(*this, ctx);
  std::vector<VarInfo> varParams =
      gazprea::builder_utils::ParamsToVarInfo(params, /*isConstDefault=*/true);
  CompleteType returnType =
      gazprea::builder_utils::ExtractReturnType(*this, ctx);
  // no body for a prototype
  auto node =
      std::make_shared<FuncPrototypeNode>(funcName, varParams, returnType);
  return node_any(std::move(node));
}
std::any
ASTBuilder::visitFunctionStat(GazpreaParser::FunctionStatContext *ctx) {
  std::string funcName = ctx->ID(0)->getText();
  std::vector<std::pair<CompleteType, std::string>> params =
      gazprea::builder_utils::ExtractParams(*this, ctx);
  std::vector<VarInfo> varParams =
      gazprea::builder_utils::ParamsToVarInfo(params, /*isConstDefault=*/true);
  CompleteType returnType =
      gazprea::builder_utils::ExtractReturnType(*this, ctx);
  std::shared_ptr<StatNode> returnStat = nullptr;
  if (ctx->stat()) {
    auto anyStat = visit(ctx->stat());
    if (anyStat.has_value()) {
      returnStat = safe_any_cast_ptr<StatNode>(anyStat);
    }
  }
  // construct FuncStatNode using VarInfo vector
  auto node = std::make_shared<FuncStatNode>(funcName, varParams, returnType,
                                             returnStat);
  return node_any(std::move(node));
}
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
  node->constant.reset();
  if(expr && expr->constant){
      auto res = gazprea::computeUnaryNumeric(*expr->constant, op);
      if(res){
        node->constant = *res;
        node->type = node->constant->type;
      }
    }
  return expr_any(std::move(node));
}
std::any ASTBuilder::visitNotExpr(GazpreaParser::NotExprContext *ctx) {
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
  auto node = std::make_shared<UnaryExpr>(op, expr);
  node->constant.reset();
  if(expr && expr->constant){
      auto res = gazprea::computeUnaryNumeric(*expr->constant, op);
      if(res){
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
    node->constant.reset();
    if(left && left->constant && right && right->constant){
      auto res = gazprea::computeBinaryComp(*left->constant, *right->constant, opText);
      if(res){
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
  node->constant.reset();
  if(left && left->constant && right && right->constant){
    auto res = gazprea::computeBinaryNumeric(*left->constant, *right->constant, opText);
    if(res){
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

  node->constant.reset();
  if(left && left->constant && right && right->constant){
    auto res = gazprea::computeBinaryNumeric(*left->constant, *right->constant, opText);
    if(res){
      node->constant = *res;
      node->type = node->constant->type;
    }
  }
  return expr_any(std::move(node));
}
std::any ASTBuilder::visitEqExpr(GazpreaParser::EqExprContext *ctx) {
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
  if (ctx->EQEQ()) {
    opText = ctx->EQEQ()->getText();
  } else if (ctx->NE()) {
    opText = ctx->NE()->getText();
  }
  auto node = std::make_shared<EqExpr>(opText, left, right);
  node->constant.reset();
  if(left && left->constant && right && right->constant){
    auto res = gazprea::computeBinaryComp(*left->constant, *right->constant, opText);
    if(res){
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
  node->constant.reset();
  if(left && left->constant && right && right->constant){
    auto res = gazprea::computeBinaryComp(*left->constant, *right->constant, opText);
    if(res){
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
  // Clear any previous constant annotation
  node->constant.reset();
  // If both children were annotated as constants, try to compute a folded value
  if (left && left->constant && right && right->constant) {
    auto res = gazprea::computeBinaryComp(*left->constant, *right->constant, opText);
    if (res) {
      node->constant = *res;
      node->type = node->constant->type;
    }
  }
  return expr_any(std::move(node));
}
std::any ASTBuilder::visitIntExpr(GazpreaParser::IntExprContext *ctx) {
  try {
    // Extract the integer literal text
    std::string text = ctx->INT()->getText();

    // Parse as 64-bit integer first to check for overflow
    long long value64 = std::stoll(text, nullptr, 10);

    // Ensure it fits within 32-bit signed integer range
    if (value64 < std::numeric_limits<int32_t>::min() ||
        value64 > std::numeric_limits<int32_t>::max()) {
      throw LiteralError(ctx->getStart()->getLine(),
                         "integer literal exceeds 32 bits");
    }

    // Convert safely to 32-bit integer
    int32_t value32 = static_cast<int32_t>(value64);

    // Create AST node
    auto node = std::make_shared<IntNode>(value32);
    node->type = CompleteType(BaseType::INTEGER);

    // ConstantValue stores a CompleteType and a std::variant payload.
    node->constant = ConstantValue(node->type, static_cast<int64_t>(value32));

    // Wrap and return as std::any
    return expr_any(std::move(node));

  } catch (const std::out_of_range &) {
    throw LiteralError(ctx->getStart()->getLine(),
                       "integer literal out of bounds");
  } catch (const std::invalid_argument &) {
    throw LiteralError(ctx->getStart()->getLine(),
                       "invalid integer literal");
  }
}
std::any ASTBuilder::visitIdExpr(GazpreaParser::IdExprContext *ctx) {
  std::string name = ctx->ID()->getText();
  // Create the IdNode and return it. Don't assign a concrete type here —
  // identifier types are resolved in the name-resolution / type-resolution
  // pass.
  auto node = std::make_shared<IdNode>(name);
  node->type = CompleteType(BaseType::STRING);
  return expr_any(std::move(node));
}
std::any ASTBuilder::visitCharExpr(GazpreaParser::CharExprContext *ctx) {
  std::string text = ctx->getText();
  char value;
  // 'c'
  if (text.length() >= 3 && text[0] == '\'' && text.back() == '\'') {
    // remove ticks
    std::string sub = text.substr(1, text.length() - 2);
    if (sub.length() == 1) {
      value = sub[0];
    } else if (sub[0] == '\\') { // '\\' is one char
      switch (sub[1]) {          // gets the next char
      case '0':
        value = '\0';
        break; // null
      case 'a':
        value = '\a';
        break; // bell
      case 'b':
        value = '\b';
        break; // backspace
      case 't':
        value = '\t';
        break; // tab
      case 'n':
        value = '\n';
        break; // line feed
      case 'r':
        value = '\r';
        break; // carriage return
      case '"':
        value = '\"';
        break; // quotation mark
      case '\'':
        value = '\'';
        break; // apostrophe
      case '\\':
        value = '\\';
        break; // backslash
      default:
        value = sub[1];
      }
    } else {
      value = sub[0];
    }
  } else { // invalid character
    throw LiteralError(ctx->getStart()->getLine(), "invalid character");
  }
  auto node = std::make_shared<CharNode>(value);
  node->type = CompleteType(BaseType::CHARACTER);
  return expr_any(std::move(node));
}

std::any ASTBuilder::visitRealExpr(GazpreaParser::RealExprContext *ctx) {
  std::string text = ctx->real()->getText();
  // apply leading zero
  double value;
  if (!text.empty() && text[0] == '.') {
    text = "0" + text;
  } else if (text.size() >= 2 && text[0] == '-' && text[1] == '.') {
    text = "-0" + text.substr(1);
  }
  try {
    value = std::stod(text); // convert to real
  } catch (const std::out_of_range &) {
    throw LiteralError(ctx->getStart()->getLine(),
                       "real literal out of bounds");
  } catch (const std::invalid_argument &) {
    throw LiteralError(ctx->getStart()->getLine(), "invalid real literal");
  }

  auto node = std::make_shared<RealNode>(value);

    // ConstantValue stores a CompleteType and a std::variant payload.
  node->type = CompleteType(BaseType::REAL);
  node->constant = ConstantValue(node->type, static_cast<double>(value));
  return expr_any(std::move(node));
}

std::any ASTBuilder::visitTrueExpr(GazpreaParser::TrueExprContext *ctx) {
  auto node = std::make_shared<TrueNode>();
  node->type = CompleteType(BaseType::BOOL);
  // Annotate with a compile-time constant true
  node->constant = ConstantValue(node->type, true);
  return expr_any(std::move(node));
}
std::any ASTBuilder::visitFalseExpr(GazpreaParser::FalseExprContext *ctx) {
  auto node = std::make_shared<FalseNode>();
  node->type = CompleteType(BaseType::BOOL);
  // Annotate with a compile-time constant false
  node->constant = ConstantValue(node->type, false);
  return expr_any(std::move(node));
}

std::any
ASTBuilder::visitTuple_literal(GazpreaParser::Tuple_literalContext *ctx) {
  std::vector<std::shared_ptr<ExprNode>> elements;
  for (auto exprCtx : ctx->expr()) {
    auto exprAny = visit(exprCtx);
    auto expr = safe_any_cast_ptr<ExprNode>(exprAny);
    if (expr)
      elements.push_back(expr);
  }
  auto node = std::make_shared<TupleLiteralNode>(elements);

  // Build the tuple CompleteType from the element expression types so
  // downstream passes have subtype information available.
  std::vector<CompleteType> elemTypes;
  elemTypes.reserve(elements.size());
  for (auto &el : elements) {
    if (el)
      elemTypes.push_back(el->type);
    else
      elemTypes.push_back(CompleteType(BaseType::UNKNOWN));
  }
  node->type = CompleteType(BaseType::TUPLE, std::move(elemTypes));
  return expr_any(std::move(node));
}

// Delegate grammar variant that wraps a tuple literal as an expression
std::any
ASTBuilder::visitTupleLitExpr(GazpreaParser::TupleLitExprContext *ctx) {
  return visit(ctx->tuple_literal());
}

std::any ASTBuilder::visitTuple_dec(GazpreaParser::Tuple_decContext *ctx) {
  // Build a CompleteType representing the tuple declaration's element types.
  std::vector<CompleteType> elemTypes;
  for (auto typeCtx : ctx->type()) {
    auto anyType = visit(typeCtx);
    if (anyType.has_value() && anyType.type() == typeid(CompleteType)) {
      elemTypes.push_back(std::any_cast<CompleteType>(anyType));
    } else {
      elemTypes.push_back(CompleteType(BaseType::UNKNOWN));
    }
  }
  return CompleteType(BaseType::TUPLE, std::move(elemTypes));
}
// LOOP (WHILE PARENLEFT expr PARENRIGHT) (block|stat) #WhileLoopBlock
std::any
ASTBuilder::visitWhileLoopBlock(GazpreaParser::WhileLoopBlockContext *ctx) {
  std::shared_ptr<ExprNode> init = nullptr;
  if (ctx->expr()) {
    auto exprAny = visit(ctx->expr());
    if (exprAny.has_value()) {
      init = safe_any_cast_ptr<ExprNode>(exprAny);
    }
  }
  std::shared_ptr<BlockNode> block = nullptr;
  std::shared_ptr<StatNode> stat = nullptr;
  // If the grammar produced a block use it, otherwise if it's a single
  // statement wrap that statement into a temporary BlockNode so LoopNode
  // always carries a BlockNode body.
  if (ctx->block()) {
    auto blockAny = visit(ctx->block());
    if (blockAny.has_value()) {
      block = safe_any_cast_ptr<BlockNode>(blockAny);
    }
  } else if (ctx->stat()) {
    auto statAny = visit(ctx->stat());
    if (statAny.has_value()) {
      stat = safe_any_cast_ptr<StatNode>(statAny);
    }
    if (stat) { // turn stats to body (blockNode)
      std::vector<std::shared_ptr<DecNode>> emptyDecs;
      std::vector<std::shared_ptr<StatNode>> stats;
      stats.push_back(stat);
      block =
          std::make_shared<BlockNode>(std::move(emptyDecs), std::move(stats));
    }
  }

  // Construct LoopNode: body (BlockNode) and condition (expr). Mark as
  // pre-check while.
  auto node = std::make_shared<LoopNode>(std::move(block), std::move(init));
  node->kind = LoopKind::While;

  return node_any(std::move(node));
}
// LOOP (block|stat) (WHILE PARENLEFT expr PARENRIGHT END)? #LoopDefault
std::any ASTBuilder::visitLoopDefault(GazpreaParser::LoopDefaultContext *ctx) {
  std::shared_ptr<BlockNode> body = nullptr;
  std::shared_ptr<ExprNode> cond = nullptr;
  bool hasCond = false;

  // extract body if present
  if (ctx->block()) {
    auto anyBody = visit(ctx->block());
    if (anyBody.has_value()) {
      body = safe_any_cast_ptr<BlockNode>(anyBody);
    }
  }

  // extract optional condition expression (for while-like loops)
  if (ctx->expr()) {
    hasCond = true;
    auto anyCond = visit(ctx->expr());
    if (anyCond.has_value()) {
      cond = safe_any_cast_ptr<ExprNode>(anyCond);
    }
  }
  // Construct LoopNode with body and optional condition, then set the kind
  auto node = std::make_shared<LoopNode>(std::move(body), std::move(cond));
  if (hasCond) {
    node->kind = LoopKind::WhilePost; // body then condition (do-while style)
  } else {
    node->kind = LoopKind::Plain; // infinite loop / no condition
  }

  return node_any(std::move(node));
}
std::any ASTBuilder::visitIf(gazprea::GazpreaParser::IfContext *ctx) {
  // Extract condition expression
  std::shared_ptr<ExprNode> cond = nullptr;
  if (ctx->expr()) {
    auto anyCond = visit(ctx->expr());
    if (anyCond.has_value()) {
      cond = safe_any_cast_ptr<ExprNode>(anyCond);
    }
  }

  // The grammar allows either a block or a single statement for the then
  // and optional else parts. Use the parse-tree positions to disambiguate
  // which block/stat corresponds to the then vs else branch.
  std::shared_ptr<BlockNode> thenBlock = nullptr;
  std::shared_ptr<BlockNode> elseBlock = nullptr;
  std::shared_ptr<StatNode> thenStat = nullptr;
  std::shared_ptr<StatNode> elseStat = nullptr;

  auto blocks = ctx->block(); // may be empty or contain up to two entries
  auto stats = ctx->stat();   // may be empty or contain up to two entries

  // Helper lambdas to visit and cast
  auto visitBlockAt = [&](size_t i) -> std::shared_ptr<BlockNode> {
    if (i >= blocks.size())
      return nullptr;
    auto anyB = visit(blocks[i]);
    if (anyB.has_value())
      return safe_any_cast_ptr<BlockNode>(anyB);
    return nullptr;
  };
  auto visitStatAt = [&](size_t i) -> std::shared_ptr<StatNode> {
    if (i >= stats.size())
      return nullptr;
    auto anyS = visit(stats[i]);
    if (anyS.has_value())
      return safe_any_cast_ptr<StatNode>(anyS);
    return nullptr;
  };

  // Determine which alternative is the 'then' part by comparing start token
  // indices of the first block/stat (if present).
  bool thenIsBlock = false;
  if (!blocks.empty() && stats.empty()) {
    thenIsBlock = true;
  } else if (blocks.empty() && !stats.empty()) {
    thenIsBlock = false;
  } else if (!blocks.empty() && !stats.empty()) {
    auto bTok = blocks[0]->getStart()->getTokenIndex();
    auto sTok = stats[0]->getStart()->getTokenIndex();
    thenIsBlock = (bTok < sTok);
  }

  if (thenIsBlock) {
    thenBlock = visitBlockAt(0);
    // else may be a second block or the first/stat(0) depending on which
    if (blocks.size() >= 2) {
      elseBlock = visitBlockAt(1);
    } else if (stats.size() >= 1) {
      // if there is a stat and it appears after the then-block, it is the
      // else branch
      if (stats[0]->getStart()->getTokenIndex() > blocks[0]->getStart()->getTokenIndex())
        elseStat = visitStatAt(0);
    }
  } else {
    thenStat = visitStatAt(0);
    if (stats.size() >= 2) {
      elseStat = visitStatAt(1);
    } else if (blocks.size() >= 1) {
      if (blocks[0]->getStart()->getTokenIndex() > stats[0]->getStart()->getTokenIndex())
        elseBlock = visitBlockAt(0);
    }
  }

  // Construct IfNode using the appropriate constructor (block-style or
  // single-statement style). Prefer block-style if thenBlock is present.
  std::shared_ptr<IfNode> node = nullptr;
  if (thenBlock) {
    node = std::make_shared<IfNode>(cond, thenBlock, elseBlock);
  } else {
    node = std::make_shared<IfNode>(cond, thenStat, elseStat);
  }

  return stat_any(std::move(node));
}

} // namespace gazprea