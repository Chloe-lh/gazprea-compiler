#include "ASTBuilderUtils.h"
#include "ASTBuilder.h"
#include "GazpreaParser.h"
#include "Types.h"
#include <any>
#include <cstddef>

namespace gazprea::builder_utils {

std::vector<VarInfo>
ParamsToVarInfo(const std::vector<std::pair<CompleteType, std::string>> &params,
                bool isConstDefault) {
  std::vector<VarInfo> out;
  out.reserve(params.size());
  for (const auto &p : params) {
    out.push_back(VarInfo{p.second, p.first, isConstDefault});
  }
  return out;
}

std::vector<std::shared_ptr<::ExprNode>> collectArgs(
    ASTBuilder &builder,
    const std::vector<gazprea::GazpreaParser::ExprContext *> &exprCtxs) {
  std::vector<std::shared_ptr<::ExprNode>> args;
  args.reserve(exprCtxs.size());
  for (auto ctx : exprCtxs) {
    if (!ctx) {
      args.push_back(nullptr);
      continue;
    }
    auto anyArg = builder.visit(ctx);
    if (anyArg.has_value()) {
      try {
        args.push_back(std::any_cast<std::shared_ptr<::ExprNode>>(anyArg));
      } catch (const std::bad_any_cast &) {
        args.push_back(nullptr);
      }
    } else {
      args.push_back(nullptr);
    }
  }
  return args;
}

static std::vector<std::pair<CompleteType, std::string>> extractParamsFromParts(
    ASTBuilder &builder,
    const std::vector<gazprea::GazpreaParser::TypeContext *> &types,
    const std::vector<antlr4::tree::TerminalNode *> &ids) {
  std::vector<std::pair<CompleteType, std::string>> out;
  size_t nTypes = types.size();
  // number of params depends on whether last type is return (callers must pass
  // only param types)
  size_t paramCount = nTypes;
  for (size_t i = 0; i < paramCount; ++i) {
    CompleteType ptype(BaseType::UNKNOWN);
    if (types[i]) {
      auto anyT = builder.visit(types[i]);
      if (anyT.has_value() && anyT.type() == typeid(CompleteType))
        ptype = std::any_cast<CompleteType>(anyT);
    }
    std::string pname;

    if (ids.size() > i && ids[i])
      pname = ids[i]->getText();
    if (pname.empty())
      pname = "_arg" + std::to_string(i);
    out.emplace_back(std::move(ptype), std::move(pname));
  }
  return out;
}

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder,
              gazprea::GazpreaParser::FunctionBlockContext *ctx) {
  // For FunctionBlock, ctx->type() has params followed by return type.
  // Pass only param types (exclude last type) and IDs (ID(0) is function name).
  size_t typeCount = ctx->type().size();
  size_t paramCount = (typeCount > 0) ? typeCount - 1 : 0;
  std::vector<GazpreaParser::TypeContext *> types;
  for (size_t i = 0; i < paramCount; ++i)
    types.push_back(ctx->type(i));
  std::vector<antlr4::tree::TerminalNode *> ids;
  auto idList = ctx->ID();
  if (idList.size() > 1) {
    for (size_t i = 1; i < idList.size() && ids.size() < paramCount; ++i) {
      ids.push_back(idList[i]);
    }
  }
  return extractParamsFromParts(builder, types, ids);
}

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder,
              gazprea::GazpreaParser::FunctionPrototypeContext *ctx) {
  // Prototype: ctx->type() contains param types (maybe) then return type last.
  // IDs: ID(0) is function name; parameter names (if present) start at ID(1).
  size_t typeCount = ctx->type().size();
  size_t paramCount = (typeCount > 0) ? typeCount - 1 : 0;
  std::vector<GazpreaParser::TypeContext *> types;
  for (size_t i = 0; i < paramCount; ++i)
    types.push_back(ctx->type(i));
  std::vector<antlr4::tree::TerminalNode *> ids;
  auto idList = ctx->ID();
  if (idList.size() > 1) {
    for (size_t i = 1; i < idList.size() && ids.size() < paramCount; ++i) {
      ids.push_back(idList[i]);
    }
  }
  return extractParamsFromParts(builder, types, ids);
}

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder,
              gazprea::GazpreaParser::FunctionBlockTupleReturnContext *ctx) {
  // tuple-return: all types in ctx->type() are parameter types (return is in
  // tuple_dec)
  size_t paramCount = ctx->type().size();
  std::vector<GazpreaParser::TypeContext *> types;
  types.reserve(paramCount);
  for (size_t i = 0; i < paramCount; ++i) {
    types.push_back(ctx->type(i));
  }
  std::vector<antlr4::tree::TerminalNode *> ids;
  auto idList = ctx->ID();
  if (idList.size() > 1) {
    for (size_t i = 1; i < idList.size() && ids.size() < paramCount; ++i) {
      ids.push_back(idList[i]);
    }
  }
  return extractParamsFromParts(builder, types, ids);
}

std::vector<std::pair<CompleteType, std::string>> ExtractParams(
    ASTBuilder &builder,
    gazprea::GazpreaParser::FunctionPrototypeTupleReturnContext *ctx) {
  // prototype tuple-return: parameter types are in ctx->type(), tuple return is
  // in ctx->tuple_dec()
  size_t paramCount = ctx->type().size();
  std::vector<GazpreaParser::TypeContext *> types;
  types.reserve(paramCount);
  for (size_t i = 0; i < paramCount; ++i) {
    types.push_back(ctx->type(i));
  }
  std::vector<antlr4::tree::TerminalNode *> ids;
  auto idList = ctx->ID();
  if (idList.size() > 1) {
    for (size_t i = 1; i < idList.size() && ids.size() < paramCount; ++i) {
      ids.push_back(idList[i]);
    }
  }
  return extractParamsFromParts(builder, types, ids);
}

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder,
              gazprea::GazpreaParser::FunctionStatContext *ctx) {
  // function stat alternative: last type in ctx->type() is the return type
  size_t typeCount = ctx->type().size();
  size_t paramCount = (typeCount > 0) ? typeCount - 1 : 0;
  std::vector<GazpreaParser::TypeContext *> types;
  types.reserve(paramCount);
  for (size_t i = 0; i < paramCount; ++i) {
    types.push_back(ctx->type(i));
  }
  std::vector<antlr4::tree::TerminalNode *> ids;
  auto idList = ctx->ID();
  if (idList.size() > 1) {
    for (size_t i = 1; i < idList.size() && ids.size() < paramCount; ++i) {
      ids.push_back(idList[i]);
    }
  }
  return extractParamsFromParts(builder, types, ids);
}

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder,
              gazprea::GazpreaParser::ProcedureContext *ctx) {
  // procedure: uses param rule with qualifier? type ID
  // Parameters are extracted from param contexts, not from separate type/ID lists
  std::vector<std::pair<CompleteType, std::string>> out;
  auto paramList = ctx->param();
  for (auto paramCtx : paramList) {
    CompleteType ptype(BaseType::UNKNOWN);
    std::string pname;
    
    if (paramCtx->type()) {
      auto anyT = builder.visit(paramCtx->type());
      if (anyT.has_value() && anyT.type() == typeid(CompleteType)) {
        ptype = std::any_cast<CompleteType>(anyT);
      }
    }
    
    if (paramCtx->ID()) {
      pname = paramCtx->ID()->getText();
    }
    if (pname.empty()) {
      pname = "_arg" + std::to_string(out.size());
    }
    
    out.emplace_back(std::move(ptype), std::move(pname));
  }
  return out;
}

// Helper to extract params with qualifiers for procedures
static std::vector<std::tuple<CompleteType, std::string, bool>> extractParamsWithQualifiers(
    ASTBuilder &builder,
    const std::vector<gazprea::GazpreaParser::ParamContext *> &params) {
  std::vector<std::tuple<CompleteType, std::string, bool>> out;
  for (auto paramCtx : params) {
    CompleteType ptype(BaseType::UNKNOWN);
    std::string pname;
    bool isConst = true; // default is const
    
    if (paramCtx->type()) {
      auto anyT = builder.visit(paramCtx->type());
      if (anyT.has_value() && anyT.type() == typeid(CompleteType)) {
        ptype = std::any_cast<CompleteType>(anyT);
      }
    }
    
    if (paramCtx->ID()) {
      pname = paramCtx->ID()->getText();
    }
    if (pname.empty()) {
      pname = "_arg" + std::to_string(out.size());
    }
    
    // Extract qualifier
    if (paramCtx->qualifier()) {
      auto qualAny = builder.visit(paramCtx->qualifier());
      if (qualAny.has_value()) {
        try {
          std::string qual = std::any_cast<std::string>(qualAny);
          isConst = (qual != "var");
        } catch (const std::bad_any_cast &) {
          // default to const
        }
      }
    }
    
    out.emplace_back(std::move(ptype), std::move(pname), isConst);
  }
  return out;
}

// helper to extract return type from type list
static CompleteType ExtractReturnTypeFromTypes(
    ASTBuilder &builder,
    const std::vector<gazprea::GazpreaParser::TypeContext *> &types) {
  if (types.empty())
    return CompleteType(BaseType::UNKNOWN);
  auto lastType = types.back();
  if (!lastType)
    return CompleteType(BaseType::UNKNOWN);

  auto anyT = builder.visit(lastType);
  if (anyT.has_value() && anyT.type() == typeid(CompleteType)) {
    return std::any_cast<CompleteType>(anyT);
  }
  return CompleteType(BaseType::UNKNOWN);
}

CompleteType
ExtractReturnType(ASTBuilder &builder,
                  gazprea::GazpreaParser::FunctionBlockContext *ctx) {
  if (!ctx)
    return CompleteType(BaseType::UNKNOWN);
  // FunctionBlockContext: ctx->type() contains param types (maybe) then return
  // type last. IDs: ID(0) is function name; parameter names (if present) start
  // at ID(1).
  size_t typeCount = ctx->type().size();
  size_t paramCount = (typeCount > 0) ? typeCount - 1 : 0;
  std::vector<GazpreaParser::TypeContext *> types;
  for (size_t i = 0; i < paramCount; ++i)
    types.push_back(ctx->type(i));
  return ExtractReturnTypeFromTypes(builder, types);
}

CompleteType
ExtractReturnType(ASTBuilder &builder,
                  gazprea::GazpreaParser::FunctionPrototypeContext *ctx) {
  if (!ctx)
    return CompleteType(BaseType::UNKNOWN);
  // FunctionProtoTypeContext: ctx->type() contains param types (maybe) then
  // return type last. IDs: ID(0) is function name; parameter names (if present)
  // start at ID(1).
  size_t typeCount = ctx->type().size();
  size_t paramCount = (typeCount > 0) ? typeCount - 1 : 0;
  std::vector<GazpreaParser::TypeContext *> types;
  for (size_t i = 0; i < paramCount; ++i)
    types.push_back(ctx->type(i));
  return ExtractReturnTypeFromTypes(builder, types);
}

CompleteType
ExtractReturnType(ASTBuilder &builder,
                  gazprea::GazpreaParser::FunctionStatContext *ctx) {
  if (!ctx)
    return CompleteType(BaseType::UNKNOWN);
  // FunctionStatContext: ctx->type() contains param types (maybe) then return
  // type last. IDs: ID(0) is function name; parameter names (if present) start
  // at ID(1).
  size_t typeCount = ctx->type().size();
  size_t paramCount = (typeCount > 0) ? typeCount - 1 : 0;
  std::vector<GazpreaParser::TypeContext *> types;
  for (size_t i = 0; i < paramCount; ++i)
    types.push_back(ctx->type(i));
  return ExtractReturnTypeFromTypes(builder, types);
}

CompleteType ExtractTupleReturnType(
    ASTBuilder &builder,
    gazprea::GazpreaParser::FunctionBlockTupleReturnContext *ctx) {
  if (!ctx)
    return CompleteType(BaseType::UNKNOWN);
  // For tuple-return variants the return is expressed with a tuple_dec node.
  if (ctx->tuple_dec()) {
    auto anyT = builder.visit(ctx->tuple_dec());
    if (anyT.has_value() && anyT.type() == typeid(CompleteType)) {
      return std::any_cast<CompleteType>(anyT);
    }
  }
  // Fallback: if tuple_dec is missing or didn't produce a CompleteType,
  // return a generic TUPLE type so downstream passes can handle it.
  return CompleteType(BaseType::TUPLE);
}

CompleteType ExtractTupleReturnType(
    ASTBuilder &builder,
    gazprea::GazpreaParser::FunctionPrototypeTupleReturnContext *ctx) {
  if (!ctx)
    return CompleteType(BaseType::UNKNOWN);
  if (ctx->tuple_dec()) {
    auto anyT = builder.visit(ctx->tuple_dec());
    if (anyT.has_value() && anyT.type() == typeid(CompleteType)) {
      return std::any_cast<CompleteType>(anyT);
    }
  }
  return CompleteType(BaseType::TUPLE);
}

// Backwards-compatible wrappers: header declares ExtractReturnType overloads
// for tuple-return contexts
CompleteType ExtractReturnType(
    ASTBuilder &builder,
    gazprea::GazpreaParser::FunctionBlockTupleReturnContext *ctx) {
  return ExtractTupleReturnType(builder, ctx);
}

CompleteType ExtractReturnType(
    ASTBuilder &builder,
    gazprea::GazpreaParser::FunctionPrototypeTupleReturnContext *ctx) {
  return ExtractTupleReturnType(builder, ctx);
}

} // namespace gazprea::builder_utils
