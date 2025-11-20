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
#include <iostream>
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
  return std::any(std::static_pointer_cast<ExprNode>(std::move(n)));
}
template <typename T> static inline std::any stat_any(std::shared_ptr<T> n) {
  return std::any(std::static_pointer_cast<StatNode>(std::move(n)));
}
template <typename T> static inline std::any dec_any(std::shared_ptr<T> n) {
  return std::any(std::static_pointer_cast<DecNode>(std::move(n)));
}

// Small helper for safely extracting shared_ptr<T> from a std::any produced
// by the builder. Avoids repeating try/catch everywhere.
template <typename T>
static inline std::shared_ptr<T> safe_any_cast_ptr(const std::any &a) {
  try {
    if (!a.has_value())
      return nullptr;
    // Exact type match
    if (a.type() == typeid(std::shared_ptr<T>))
      return std::any_cast<std::shared_ptr<T>>(a);
    // Upcast case: value stored as std::shared_ptr<ASTNode>
    if (a.type() == typeid(std::shared_ptr<ASTNode>)) {
      auto base = std::any_cast<std::shared_ptr<ASTNode>>(a);
      return std::dynamic_pointer_cast<T>(base);
    }
    // Common families also derive from ASTNode; try those too
    if (a.type() == typeid(std::shared_ptr<ExprNode>)) {
      auto base = std::any_cast<std::shared_ptr<ExprNode>>(a);
      return std::dynamic_pointer_cast<T>(base);
    }
    if (a.type() == typeid(std::shared_ptr<StatNode>)) {
      auto base = std::any_cast<std::shared_ptr<StatNode>>(a);
      return std::dynamic_pointer_cast<T>(base);
    }
    if (a.type() == typeid(std::shared_ptr<DecNode>)) {
      auto base = std::any_cast<std::shared_ptr<DecNode>>(a);
      return std::dynamic_pointer_cast<T>(base);
    }
  } catch (const std::bad_any_cast &) {
    // fall through
  }
  return nullptr;
}

std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
  // if (getenv("GAZ_DEBUG")) std::cerr << "visiting file";
  std::vector<std::shared_ptr<ASTNode>> nodes;
  for (auto child : ctx->children) {
    auto anyNode = visit(child);
    if (anyNode.has_value()) {
      std::shared_ptr<ASTNode> node = nullptr;
      // Try casting to ASTNode first
      try {
        node = std::any_cast<std::shared_ptr<ASTNode>>(anyNode);
      } catch (const std::bad_any_cast &) {
        // Try casting to StatNode and upcast
        try {
          auto statNode = std::any_cast<std::shared_ptr<StatNode>>(anyNode);
          node = std::static_pointer_cast<ASTNode>(statNode);
        } catch (const std::bad_any_cast &) {
          // Try casting to DecNode and upcast
          try {
            auto decNode = std::any_cast<std::shared_ptr<DecNode>>(anyNode);
            node = std::static_pointer_cast<ASTNode>(decNode);
          } catch (const std::bad_any_cast &) {
            // Try casting to ExprNode and upcast (unlikely but possible)
            try {
              auto exprNode = std::any_cast<std::shared_ptr<ExprNode>>(anyNode);
              node = std::static_pointer_cast<ASTNode>(exprNode);
            } catch (const std::bad_any_cast &) {
              // Skip invalid node
            }
          }
        }
      }
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
std::any ASTBuilder::visitTupleTypeCastExpr(
    GazpreaParser::TupleTypeCastExprContext *ctx) {
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
std::any
ASTBuilder::visitTypeCastExpr(GazpreaParser::TypeCastExprContext *ctx) {
  // Determine the target type (returns a CompleteType from visitType)
  CompleteType target = CompleteType(BaseType::UNKNOWN);
  auto tctx = ctx->type();
  if (tctx) {
    auto anyType = visit(tctx);
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
  // Preserve alias name text for casts like as<AliasName>(expr)
  if (tctx && tctx->ID()) {
    node->targetAliasName = tctx->ID()->getText();
    node->targetType = CompleteType(BaseType::UNKNOWN);
  }
  return expr_any(std::move(node));
}
/*
dec
: qualifier? (builtin_type ID | ID ID) (EQ expr)? END   #ExplicitTypedDec
| qualifier ID EQ expr END                              #InferredTypeDec
| qualifier? tuple_dec ID (EQ expr)? END                #TupleTypedDec
*/
std::any
ASTBuilder::visitExplicitTypedDec(GazpreaParser::ExplicitTypedDecContext *ctx) {
  // qualifier (optional) - defaults to "const" if not provided
  std::string qualifier = "const";
  if (ctx->qualifier()) {
    auto qualAny = visit(ctx->qualifier());
    if (qualAny.has_value()) {
      try {
        qualifier = std::any_cast<std::string>(qualAny);
      } catch (const std::bad_any_cast &) {
        qualifier = "const";
      }
    }
  }

  std::shared_ptr<TypeAliasNode> typeAlias;
  std::string id;

  // builtin_type ID case
  if (ctx->builtin_type()) {
    auto bt = ctx->builtin_type();
    if (bt->BOOLEAN())
      typeAlias =
          std::make_shared<TypeAliasNode>("", CompleteType(BaseType::BOOL));
    else if (bt->CHARACTER())
      typeAlias = std::make_shared<TypeAliasNode>(
          "", CompleteType(BaseType::CHARACTER));
    else if (bt->INTEGER())
      typeAlias =
          std::make_shared<TypeAliasNode>("", CompleteType(BaseType::INTEGER));
    else if (bt->REAL())
      typeAlias =
          std::make_shared<TypeAliasNode>("", CompleteType(BaseType::REAL));
    else if (bt->STRING())
      typeAlias =
          std::make_shared<TypeAliasNode>("", CompleteType(BaseType::STRING));
    else
      typeAlias =
          std::make_shared<TypeAliasNode>("", CompleteType(BaseType::UNKNOWN));
    id = ctx->ID(0)->getText();
  } else {
    // alias ID ID case: ID(0) alias name, ID(1) variable name
    std::string aliasName = ctx->ID(0)->getText();
    id = ctx->ID(1)->getText();
    typeAlias = std::make_shared<TypeAliasNode>(
        aliasName, CompleteType(BaseType::UNKNOWN));
  }

  // optional initializer
  std::shared_ptr<ExprNode> init = nullptr;
  if (ctx->expr()) {
    auto exprAny = visit(ctx->expr());
    if (exprAny.has_value())
      init = safe_any_cast_ptr<ExprNode>(exprAny);
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
  auto node = std::make_shared<InferredDecNode>(id, qualifier, expr);
  return node_any(std::move(node));
}
std::any
ASTBuilder::visitTupleTypedDec(GazpreaParser::TupleTypedDecContext *ctx) {
  std::string qualifier = "const"; // default to const
  if (ctx->qualifier()) {
    auto qualAny = visit(ctx->qualifier());
    if (qualAny.has_value()) {
      try {
        qualifier = std::any_cast<std::string>(qualAny);
      } catch (const std::bad_any_cast &) {
        qualifier = "const";
      }
    }
  }

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

  auto node = std::make_shared<TupleTypedDecNode>(id, qualifier, tupleType);
  node->init = init;
  return node_any(std::move(node));
}
std::any
ASTBuilder::visitTupleAccessExpr(GazpreaParser::TupleAccessExprContext *ctx) {
  // tuple_access: ID '.' INT | TUPACCESS
  auto ta = ctx->tuple_access();
  std::string tupleName = "";
  int index = 0;
  if (ta) {
    if (ta->TUPACCESS()) {
      // token form: name.index
      std::string text = ta->TUPACCESS()->getText();
      auto pos = text.find('.');
      if (pos != std::string::npos) {
        tupleName = text.substr(0, pos);
        try {
          index = std::stoi(text.substr(pos + 1));
        } catch (...) {
          index = 0;
        }
      }
    } else {
      if (ta->ID())
        tupleName = ta->ID()->getText();
      if (ta->INT()) {
        try {
          index = std::stoi(ta->INT()->getText());
        } catch (const std::exception &) {
          index = 0;
        }
      }
    }
  }
  auto node = std::make_shared<TupleAccessNode>(tupleName, index);
  return expr_any(std::move(node));
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
  if (ctx->STRING())
    return CompleteType(BaseType::STRING);
  if (ctx->ID())
    // Store the alias name, but leaves the type as BaseType::UNRESOLVED, which
    // will be resolved during semantic analysis
    return CompleteType(ctx->ID()->getText());
  if (ctx->INTEGER())
    return CompleteType(BaseType::INTEGER);
  if (ctx->REAL())
    return CompleteType(BaseType::REAL);
  if (ctx->CHARACTER())
    return CompleteType(BaseType::CHARACTER);

  throw std::runtime_error(
      "ASTBuilder::visitType: FATAL: Type with no known case.");
}
std::any
ASTBuilder::visitBasicTypeAlias(GazpreaParser::BasicTypeAliasContext *ctx) {
  // Grammar: TYPEALIAS type ID
  std::string referenced = ctx->type()->getText();
  std::string aliasName = ctx->ID()->getText();

  CompleteType aliasedType(BaseType::UNKNOWN);
  if (referenced == "integer")
    aliasedType = CompleteType(BaseType::INTEGER);
  else if (referenced == "real")
    aliasedType = CompleteType(BaseType::REAL);
  else if (referenced == "boolean")
    aliasedType = CompleteType(BaseType::BOOL);
  else if (referenced == "character")
    aliasedType = CompleteType(BaseType::CHARACTER);
  else
    throw std::runtime_error("aliasing an alias not implemented");

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
std::any ASTBuilder::visitDestructAssignStat(
    GazpreaParser::DestructAssignStatContext *ctx) {
  std::vector<std::string> names;
  auto idList = ctx->ID();
  names.reserve(idList.size());
  for (auto *idTok : idList) {
    if (idTok) {
      names.push_back(idTok->getText());
    }
  }
  auto exprAny = visit(ctx->expr());
  auto expr = safe_any_cast_ptr<ExprNode>(exprAny);
  auto node = std::make_shared<DestructAssignStatNode>(std::move(names), expr);
  return stat_any(std::move(node));
}
std::any ASTBuilder::visitTupleAccessAssignStat(
    GazpreaParser::TupleAccessAssignStatContext *ctx) {
  if (!ctx || !ctx->tuple_access() || !ctx->expr()) {
    throw std::runtime_error("visitTupleAccessAssignStat: invalid context");
  }

  // Build the LHS TupleAccessNode directly from the tuple_access rule,
  // mirroring visitTupleAccessExpr.
  GazpreaParser::Tuple_accessContext *ta = ctx->tuple_access();
  std::string tupleName;
  int index = 0;

  if (ta) {
    if (ta->TUPACCESS()) {
      std::string text = ta->TUPACCESS()->getText();
      auto pos = text.find('.');
      if (pos != std::string::npos) {
        tupleName = text.substr(0, pos);
        try {
          index = std::stoi(text.substr(pos + 1));
        } catch (...) {
          index = 0;
        }
      }
    } else {
      if (ta->ID())
        tupleName = ta->ID()->getText();
      if (ta->INT()) {
        try {
          index = std::stoi(ta->INT()->getText());
        } catch (const std::exception &) {
          index = 0;
        }
      }
    }
  }

  auto lhs = std::make_shared<TupleAccessNode>(tupleName, index);

  auto rhsAny = visit(ctx->expr());
  auto rhs = safe_any_cast_ptr<ExprNode>(rhsAny);

  auto node = std::make_shared<TupleAccessAssignStatNode>(std::move(lhs),
                                                          std::move(rhs));
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
  auto args = gazprea::builder_utils::collectArgs(*this, ctx->expr());
  auto node = std::make_shared<FuncCallExpr>(funcName, std::move(args));
  return expr_any(std::move(node));
}
//  CALL ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT END  #CallStat
std::any ASTBuilder::visitCallStat(GazpreaParser::CallStatContext *ctx) {
  if (!ctx || !ctx->ID()) {
    throw std::runtime_error("visitCallStat: invalid call statement context");
  }

  std::string funcName = ctx->ID()->getText();

  // Safely collect arguments
  std::vector<GazpreaParser::ExprContext *> exprCtxs;
  if (ctx->expr().size() > 0) {
    exprCtxs.reserve(ctx->expr().size());
    for (auto exprCtx : ctx->expr()) {
      if (exprCtx) {
        exprCtxs.push_back(exprCtx);
      }
    }
  }

  auto args = gazprea::builder_utils::collectArgs(*this, exprCtxs);

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

  // Check for regular expression first
  if (ctx->expr()) {
    auto anyExpr = visit(ctx->expr());
    if (anyExpr.has_value()) {
      try {
        expr = std::any_cast<std::shared_ptr<ExprNode>>(anyExpr);
      } catch (const std::bad_any_cast &) {
        expr = nullptr;
      }
    }
  }
  // Check for tuple_access (grammar has: tuple_access '->' STD_OUTPUT)
  else if (ctx->tuple_access()) {
    // Manually parse tuple_access since there's no visitor method for it
    auto ta = ctx->tuple_access();
    std::string tupleName = "";
    int index = 0;
    if (ta) {
      if (ta->TUPACCESS()) {
        // token form: name.index
        std::string text = ta->TUPACCESS()->getText();
        auto pos = text.find('.');
        if (pos != std::string::npos) {
          tupleName = text.substr(0, pos);
          try {
            index = std::stoi(text.substr(pos + 1));
          } catch (...) {
            index = 0;
          }
        }
      } else {
        if (ta->ID())
          tupleName = ta->ID()->getText();
        if (ta->INT()) {
          try {
            index = std::stoi(ta->INT()->getText());
          } catch (const std::exception &) {
            index = 0;
          }
        }
      }
    }
    auto tupleAccessNode = std::make_shared<TupleAccessNode>(tupleName, index);
    expr = std::static_pointer_cast<ExprNode>(tupleAccessNode);
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

std::any ASTBuilder::visitProcedurePrototype(
    GazpreaParser::ProcedurePrototypeContext *ctx) {
  std::string procName = ctx->ID()->getText();
  auto tuples =
      gazprea::builder_utils::ExtractParamsWithQualifiers(*this, ctx->param());
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
      gazprea::builder_utils::ExtractParamsWithQualifiers(*this, ctx->param());
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
  if (ctx->expr()) {
    auto anyExpr = visit(ctx->expr());
    auto exprNode = safe_any_cast_ptr<ExprNode>(anyExpr);
    if (exprNode) {
      auto retNode = std::make_shared<ReturnStatNode>(exprNode);
      returnStat = retNode;
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
    throw LiteralError(ctx->getStart()->getLine(), "invalid integer literal");
  }
}
std::any ASTBuilder::visitIdExpr(GazpreaParser::IdExprContext *ctx) {
  if (!ctx || !ctx->ID()) {
    throw std::runtime_error(
        "visitIdExpr: invalid identifier expression context");
  }

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
std::any ASTBuilder::visitStringExpr(GazpreaParser::StringExprContext *ctx) {
  std::string text = ctx->STRING_LIT()->getText();
  // Strip the surrounding quotes and unescape minimal sequences handled in CHAR
  std::string out;
  out.reserve(text.size());
  // remove leading and trailing quotes if present
  size_t i = 0, n = text.size();
  if (n >= 2 && text.front() == '"' && text.back() == '"') {
    i = 1;
    n -= 1;
  }
  while (i < n) {
    char c = text[i++];
    if (c == '\\' && i < n) {
      char e = text[i++];
      switch (e) {
      case '0':
        out.push_back('\0');
        break;
      case 'a':
        out.push_back('\a');
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case '"':
        out.push_back('"');
        break;
      case '\'':
        out.push_back('\'');
        break;
      case '\\':
        out.push_back('\\');
        break;
      default:
        out.push_back(e);
        break;
      }
    } else {
      out.push_back(c);
    }
  }
  auto node = std::make_shared<StringNode>(out);
  return expr_any(std::move(node));
}

std::any ASTBuilder::visitParenExpr(GazpreaParser::ParenExprContext *ctx) {
  std::shared_ptr<ExprNode> inner = nullptr;
  if (ctx->expr()) {
    auto anyInner = visit(ctx->expr());
    if (anyInner.has_value()) {
      inner = safe_any_cast_ptr<ExprNode>(anyInner);
    }
  }
  auto node = std::make_shared<ParenExpr>(inner);
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

// if: IF PARENLEFT expr PARENRIGHT (block|stat) (ELSE (block|stat))?;
std::any ASTBuilder::visitIfStat(gazprea::GazpreaParser::IfStatContext *ctx) {
  auto ifCtx = ctx->if_stat();
  if (!ifCtx) {
    return nullptr;
  }

  // Visit the condition and create the IfNode using the new constructor.
  auto cond = safe_any_cast_ptr<ExprNode>(visit(ifCtx->expr()));
  auto node = std::make_shared<IfNode>(cond);

  // The ANTLR grammar guarantees the 'then' branch is the first block or first
  // statement.
  if (!ifCtx->block().empty()) {
    node->thenBlock = safe_any_cast_ptr<BlockNode>(visit(ifCtx->block(0)));
  } else {
    node->thenStat = safe_any_cast_ptr<StatNode>(visit(ifCtx->stat(0)));
  }

  // Determine and visit the 'else' branch, if it exists.
  if (ifCtx->ELSE()) {
    bool thenWasBlock = (node->thenBlock != nullptr);

    if (thenWasBlock) {
      // If 'then' was a block, 'else' can be the second block or the first
      // statement.
      if (ifCtx->block().size() > 1) {
        node->elseBlock = safe_any_cast_ptr<BlockNode>(visit(ifCtx->block(1)));
      } else if (!ifCtx->stat().empty()) {
        node->elseStat = safe_any_cast_ptr<StatNode>(visit(ifCtx->stat(0)));
      }
    } else { // 'then' was a statement
      // If 'then' was a statement, 'else' can be the first block or the second
      // statement.
      if (!ifCtx->block().empty()) {
        node->elseBlock = safe_any_cast_ptr<BlockNode>(visit(ifCtx->block(0)));
      } else if (ifCtx->stat().size() > 1) {
        node->elseStat = safe_any_cast_ptr<StatNode>(visit(ifCtx->stat(1)));
      }
    }
  }

  return node_any(std::move(node));
}

std::any ASTBuilder::visitLoopStat(GazpreaParser::LoopStatContext *ctx) {

  // TODO: fix this
  auto loopCtx = ctx->loop_stat();
  if (!loopCtx) {
    return nullptr;
  }
  return visit(loopCtx);
}

} // namespace gazprea
