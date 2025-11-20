#pragma once
#include "AST.h"
#include "ASTBuilder.h"
#include "Scope.h"
#include "Types.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

// We need the generated parser definition here because we reference nested
// types like GazpreaParser::ExprContext in function signatures.
#include "GazpreaParser.h"

namespace gazprea {
class ASTBuilder;
}

namespace gazprea::builder_utils {
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

} // namespace gazprea::builder_utils
