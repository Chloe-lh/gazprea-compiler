#include "ASTBuilder.h"
#include "ASTBuilderUtils.h"
#include "AST.h"

namespace gazprea {

//  TupleTypeAliasNode(const std::string& aliasName, CompleteType tupleType);
// TYPEALIAS tuple_dec ID
std::any ASTBuilder::visitTupleTypeAlias(GazpreaParser::TupleTypeAliasContext *ctx) {
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
  setLocationFromCtx(node, ctx);
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
        index = std::stoi(text.substr(pos + 1));
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
  setLocationFromCtx(node, ctx);
  return expr_any(std::move(node));
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

  auto node = std::make_shared<TupleTypedDecNode>(qualifier, id,tupleType);
  setLocationFromCtx(node, ctx);
  node->init = init;
  node->qualifier = qualifier;
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
  setLocationFromCtx(node, ctx);
  return expr_any(std::move(node));
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
  setLocationFromCtx(node, ctx);
  return stat_any(std::move(node));
}
std::any ASTBuilder::visitBreakStat(GazpreaParser::BreakStatContext *ctx) {
  auto node = std::make_shared<BreakStatNode>();
  setLocationFromCtx(node, ctx);
  return stat_any(std::move(node));
}
std::any
ASTBuilder::visitContinueStat(GazpreaParser::ContinueStatContext *ctx) {
  auto node = std::make_shared<ContinueStatNode>();
  setLocationFromCtx(node, ctx);
  return stat_any(std::move(node));
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
  setLocationFromCtx(node, ctx);

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

} // namespace gazprea