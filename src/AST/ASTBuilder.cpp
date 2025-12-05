#include "ASTBuilder.h"
#include "AST.h"
#include "ASTBuilderUtils.h"
#include "CompileTimeExceptions.h"
#include "ParserRuleContext.h"
#include "ConstantHelpers.h"
#include "GazpreaParser.h"
#include "Types.h"
#include <any>
#include <memory>
#include <stdexcept>
#include <stdlib.h>

namespace gazprea {

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
          // setLocationFromCtx(node, ctx);
        } catch (const std::bad_any_cast &) {
          // Try casting to DecNode and upcast
          try {
            auto decNode = std::any_cast<std::shared_ptr<DecNode>>(anyNode);
            node = std::static_pointer_cast<ASTNode>(decNode);
            // setLocationFromCtx(node, ctx);
          } catch (const std::bad_any_cast &) {
            // Try casting to ExprNode and upcast (unlikely but possible)
              try {
              auto exprNode = std::any_cast<std::shared_ptr<ExprNode>>(anyNode);
              node = std::static_pointer_cast<ASTNode>(exprNode);
              // setLocationFromCtx(node, ctx);
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
  setLocationFromCtx(node, ctx);
  return node_any(std::move(node));
}
std::any ASTBuilder::visitBuiltin_type(GazpreaParser::Builtin_typeContext *ctx) {
    // 1. Base scalar type 
    CompleteType elem(BaseType::UNKNOWN);

    if (ctx->BOOLEAN())   elem = CompleteType(BaseType::BOOL);
    if (ctx->STRING())    elem = CompleteType(BaseType::STRING);
    if (ctx->INTEGER())   elem = CompleteType(BaseType::INTEGER);
    if (ctx->REAL())      elem = CompleteType(BaseType::REAL);
    if (ctx->CHARACTER()) elem = CompleteType(BaseType::CHARACTER);

    // 2. Vector<T> case: VECTOR '<' type '>'
    if (ctx->VECTOR()) {
      auto type = visit(ctx->type());
      CompleteType elem = std::any_cast<CompleteType>(type);
      return CompleteType(BaseType::VECTOR, elem, {-1});
    }

    // 3. Scalar type (no array size)
    if (!ctx->size()) {
        // Return the scalar CompleteType (elem) wrapped in std::any.
        return elem;
    }

    // 4. Array dimensions encoded directly into CompleteType::dims
    // Grammar: size : '[' (INT|MULT) ']' ('[' (INT|MULT) ']')?
    auto *sizectx = ctx->size();
    std::vector<int> dims;
    for (auto child : sizectx->children) {
        auto *t = dynamic_cast<antlr4::tree::TerminalNode*>(child);
        if (!t) continue;

        int token = t->getSymbol()->getType();
        if (token == GazpreaParser::INT) {
            int64_t val = std::stoll(t->getText());
            dims.push_back(static_cast<int>(val));
        } else if (token == GazpreaParser::MULT) {
            // '*' inferred size, represent with -1
            dims.push_back(-1);
        }
    }
    if(dims.size()==2){
      CompleteType arr(BaseType::ARRAY);
    }
    CompleteType arr(BaseType::ARRAY);
    arr.subTypes.push_back(elem);        // element type as single subtype
    arr.dims = std::move(dims);          // save dimensions
    return arr;
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
  setLocationFromCtx(node, ctx);
  return node_any(std::move(node));
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
  setLocationFromCtx(node, ctx);
  // Preserve alias name text for casts like as<AliasName>(expr)
  if (tctx && tctx->ID()) {
    node->targetAliasName = tctx->ID()->getText();
    node->targetType = CompleteType(BaseType::UNKNOWN);
  }
  return expr_any(std::move(node));
}
/*
dec
  Also has support for arrays 
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

  // optional initializer (evaluate early so array-case can use it)
  std::shared_ptr<ExprNode> init = nullptr;
  std::any exprAny;
  if (ctx->expr()) {
    auto anyExpr = visit(ctx->expr());
    if (anyExpr.has_value()) {
    init = safe_any_cast_ptr<ExprNode>(anyExpr);
    }
  }

  // builtin_type ID case
  if (ctx->builtin_type()) {
    // visiting builtin_type now always returns a CompleteType (scalar/array/vector)
    auto anyType = visit(ctx->builtin_type());
    // variable name is always the ID after the type
    if (ctx->ID().size() > 0) {
      id = ctx->ID(0)->getText();
    }

    if (!anyType.has_value()) {
      throw std::runtime_error("ASTBuilder::visitExplicitTypedDec: builtin_type produced no type for '" + id + "'.");
    }
    if (anyType.type() != typeid(CompleteType)) {
      throw std::runtime_error("ASTBuilder::visitExplicitTypedDec: builtin_type returned unsupported node type for '" + id + "'.");
    }

    CompleteType ct = std::any_cast<CompleteType>(anyType);


    // Arrays and vectors bundled into same node -
    // Distinguished by `CompleteType::baseType`
    if (ct.baseType == BaseType::ARRAY || ct.baseType == BaseType::VECTOR) {
      auto arrDec = std::make_shared<ArrayTypedDecNode>(qualifier, id, ct, init);
      setLocationFromCtx(arrDec, ctx);
      return node_any(std::move(arrDec));
    }

    if (ct.baseType == BaseType:: MATRIX) {
      throw std::runtime_error(
          "ASTBuilder::visitExplicitTypedDec: vector and matrix not supported yet");
    }

    // All other builtin types (scalar/tuple/struct/string) are represented
    // via a TypeAliasNode that simply carries the concrete CompleteType.
    // The empty alias name ensures semantic analysis does not treat it as a
    // user-defined alias.
    typeAlias = std::make_shared<TypeAliasNode>(std::string(""), ct);
  } else {
    // alias ID ID case: ID(0) alias name, ID(1) variable name
    std::string aliasName = ctx->ID(0)->getText();
    id = ctx->ID(1)->getText();
    typeAlias = std::make_shared<TypeAliasNode>(
        aliasName, CompleteType(BaseType::UNKNOWN));
  }
  
  auto node = std::make_shared<TypedDecNode>(id, typeAlias, qualifier, init);
  setLocationFromCtx(node, ctx);
  return node_any(std::move(node));
}

std::any ASTBuilder::visitInferredTypeDec(GazpreaParser::InferredTypeDecContext *ctx) {
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
  setLocationFromCtx(node, ctx);
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
  // 1. Handle builtin types (everything except tuple + struct + aliases)
  if (ctx->builtin_type()) {
    return visit(ctx->builtin_type());
  }

  // 2. Handle type aliases - will be resolved to actual types in semantic analysis
  if (ctx->ID())
    return CompleteType(ctx->ID()->getText());

  // 3. handle tuples + structs
  if (ctx->tuple_dec()) 
    return visit(ctx->tuple_dec());
  if (ctx->struct_dec())
    return visit(ctx->struct_dec());

  throw std::runtime_error(
      "ASTBuilder::visitType: FATAL: Type ctx '" + ctx->getText() + "' with no known case.");
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
    throw std::runtime_error("aliasing an alias not implemented for alias '" + aliasName + "' with type '" + referenced + "'.");

  auto node = std::make_shared<TypeAliasDecNode>(aliasName, aliasedType);
  setLocationFromCtx(node, ctx);
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
  setLocationFromCtx(node, ctx);
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
  setLocationFromCtx(node, ctx);
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
  setLocationFromCtx(node, ctx);
  return stat_any(std::move(node));
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

  auto args = collectArgs(*this, exprCtxs);

  // Build an expression-level call node and wrap it in a CallStatNode
  auto callExpr = std::make_shared<FuncCallExprOrStructLiteral>(funcName, std::move(args));
  auto node = std::make_shared<CallStatNode>(callExpr);
  setLocationFromCtx(node, ctx);
  return stat_any(std::move(node));
}

std::any ASTBuilder::visitInputStat(GazpreaParser::InputStatContext *ctx) {
  std::string id = ctx->ID()->getText();
  auto node = std::make_shared<InputStatNode>(id);
  setLocationFromCtx(node, ctx);
  return stat_any(std::move(node));
}

std::any ASTBuilder::visitOutputStat(GazpreaParser::OutputStatContext *ctx) {
  std::shared_ptr<ExprNode> expr = nullptr;
  if (!ctx->expr()) throw std::runtime_error("visitOutputStat: No expr found");
  
  auto anyExpr = visit(ctx->expr());
  if (anyExpr.has_value()) {
    expr = safe_any_cast_ptr<ExprNode>(anyExpr);
  }

  auto node = std::make_shared<OutputStatNode>(expr);
  setLocationFromCtx(node, ctx);
  return stat_any(std::move(node));
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

std::any ASTBuilder::visitIntExpr(GazpreaParser::IntExprContext *ctx) {
    // Extract text
    std::string text = ctx->INT()->getText();

    // Check range using manual parsing (better than std::stoll)
    long long value64;

    try {
        value64 = std::stoll(text, nullptr, 10);
    } catch (const std::out_of_range &) {
        throw LiteralError(ctx->getStart()->getLine(),
                           "integer literal out of bounds");
    } catch (const std::invalid_argument &) {
        throw LiteralError(ctx->getStart()->getLine(),
                           "invalid integer literal");
    }

    // Check 32-bit range
    if (value64 < std::numeric_limits<int32_t>::min() ||
        value64 > std::numeric_limits<int32_t>::max()) {
        throw LiteralError(ctx->getStart()->getLine(),
                           "integer literal exceeds 32 bits");
    }

    // Convert safely
    int32_t v32 = static_cast<int32_t>(value64);

    auto node = std::make_shared<IntNode>(v32);
    setLocationFromCtx(node, ctx);
    node->type = CompleteType(BaseType::INTEGER);
    node->value = v32;
    node->constant = ConstantValue(node->type, (int64_t)v32);

    return expr_any(std::move(node));
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
  setLocationFromCtx(node, ctx);
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
  setLocationFromCtx(node, ctx);
  node->type = CompleteType(BaseType::CHARACTER);
  return expr_any(std::move(node));
}


std::any ASTBuilder::visitStdInputExpr(GazpreaParser::StdInputExprContext *ctx) {
  // Treat std_input as a dummy integer 0
  auto node = std::make_shared<IntNode>(0);
  setLocationFromCtx(node, ctx);
  node->type = CompleteType(BaseType::INTEGER); // Matches input_stream alias
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
  setLocationFromCtx(node, ctx);
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
  setLocationFromCtx(node, ctx);
  return expr_any(std::move(node));
}

std::any ASTBuilder::visitRealExpr(GazpreaParser::RealExprContext *ctx) {
  std::string text = ctx->real()->getText();
  // apply leading zero
  double value = 0; // initialize to zero
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
  setLocationFromCtx(node, ctx);

  // ConstantValue stores a CompleteType and a std::variant payload.
  node->type = CompleteType(BaseType::REAL);
  node->constant = ConstantValue(node->type, static_cast<double>(value));
  return expr_any(std::move(node));
}

std::any ASTBuilder::visitTrueExpr(GazpreaParser::TrueExprContext *ctx) {
  auto node = std::make_shared<TrueNode>();
  setLocationFromCtx(node, ctx);
  node->type = CompleteType(BaseType::BOOL);
  // Annotate with a compile-time constant true
  node->constant = ConstantValue(node->type, true);
  return expr_any(std::move(node));
}
std::any ASTBuilder::visitFalseExpr(GazpreaParser::FalseExprContext *ctx) {
  auto node = std::make_shared<FalseNode>();
  setLocationFromCtx(node, ctx);
  node->type = CompleteType(BaseType::BOOL);
  // Annotate with a compile-time constant false
  node->constant = ConstantValue(node->type, false);
  return expr_any(std::move(node));
}
} // namespace gazprea
