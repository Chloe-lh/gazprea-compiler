#pragma once
#include "AST.h"
#include "ASTBuilder.h"
#include "Scope.h"
#include "Types.h"
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// We need the generated parser definition here because we reference nested
// types like GazpreaParser::ExprContext in function signatures.
#include "GazpreaParser.h"
#include <any>

namespace gazprea {
class ASTBuilder;
}

namespace gazprea {
std::vector<VarInfo>
ParamsToVarInfo(const std::vector<std::pair<CompleteType, std::string>> &params,
                bool isConstDefault);

// Collect argument expressions into ExprNode shared_ptrs using the provided
// ASTBuilder (calls builder.visit on each expr). Returns a vector whose
// size matches the input exprCtxs; elements may be nullptr if a visit failed.
std::vector<std::shared_ptr<ExprNode>>
collectArgs(ASTBuilder &builder,
            const std::vector<GazpreaParser::ExprContext *> &exprCtxs);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder, GazpreaParser::FunctionBlockContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder,
              GazpreaParser::ProcedureBlockContext *ctx);

// Extract parameters with qualifiers (const/var) from a list of `param` nodes.
// Each tuple is (type, name, isConst).
std::vector<std::tuple<CompleteType, std::string, bool>>
ExtractParamsWithQualifiers(
    ASTBuilder &builder,
    const std::vector<GazpreaParser::ParamContext *> &params);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder,
              GazpreaParser::FunctionPrototypeContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder,
              GazpreaParser::FunctionBlockTupleReturnContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder,
              GazpreaParser::FunctionPrototypeTupleReturnContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(ASTBuilder &builder, GazpreaParser::FunctionStatContext *ctx);

CompleteType ExtractReturnType(ASTBuilder &builder,
                               GazpreaParser::FunctionBlockContext *ctx);
CompleteType ExtractReturnType(ASTBuilder &builder,
                               GazpreaParser::FunctionPrototypeContext *ctx);
CompleteType ExtractReturnType(ASTBuilder &builder,
                               GazpreaParser::FunctionStatContext *ctx);
CompleteType
ExtractReturnType(ASTBuilder &builder,
                  GazpreaParser::FunctionBlockTupleReturnContext *ctx);
CompleteType
ExtractReturnType(ASTBuilder &builder,
                  GazpreaParser::FunctionPrototypeTupleReturnContext *ctx);

// Convenience helper: convert ExtractParams output to VarInfo vector
std::vector<VarInfo>
ParamsToVarInfo(const std::vector<std::pair<CompleteType, std::string>> &params,
                bool isConstDefault = true);

// Set source location (line/column) on an AST node from a parser context.
// This helper is used by many visitor implementations to attach a 1-based
// source line number to nodes.
void setLocationFromCtx(std::shared_ptr<ASTNode> node, antlr4::ParserRuleContext *ctx);

// Helper to safely extract std::shared_ptr<T> from a std::any produced by
// ASTBuilder::visit(...) calls. This centralizes the logic so all builder
// translation units can use the same implementation.
template <typename T>
static inline std::shared_ptr<T> safe_any_cast_ptr(const std::any &a) {
    try {
        if (!a.has_value())
            return nullptr;
        if (a.type() == typeid(std::shared_ptr<T>))
            return std::any_cast<std::shared_ptr<T>>(a);
        if (a.type() == typeid(std::shared_ptr<ASTNode>)) {
            auto base = std::any_cast<std::shared_ptr<ASTNode>>(a);
            return std::dynamic_pointer_cast<T>(base);
        }
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
        if (a.type() == typeid(std::shared_ptr<LiteralExprNode>)){
            auto base = std::any_cast<std::shared_ptr<LiteralExprNode>>(a);
            return std::dynamic_pointer_cast<T>(base);
        }
    } catch (const std::bad_any_cast &) {
        // fall through
    }
    return nullptr;
}
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
template <typename T> static inline std::any litexpr_any(std::shared_ptr<T> n) {
  return std::any(std::static_pointer_cast<LiteralExprNode>(std::move(n)));
}

} // namespace gazprea::builder_utils

