#pragma once
#include "Types.h"
#include "Scope.h"
#include <vector>
#include <string>
#include <utility>

// Forward-declarex generated parser and its context classes
// include GazpreaParser giving issues 
class GazpreaParser {
public:
	class FunctionBlockContext;
	class ProcedureContext;
	class FunctionStatContext;
    class FunctionPrototypeContext;
    class FunctionBlockTupleReturnContext;
    class FunctionPrototypeTupleReturnContext;
};

namespace gazprea { class ASTBuilder; }
namespace gazprea { namespace builder_utils {

std::vector<VarInfo> ParamsToVarInfo(const std::vector<std::pair<CompleteType, std::string>>& params,
                                     bool isConstDefault);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(gazprea::ASTBuilder &builder, GazpreaParser::FunctionBlockContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(gazprea::ASTBuilder &builder, GazpreaParser::ProcedureContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(gazprea::ASTBuilder &builder, GazpreaParser::FunctionPrototypeContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(gazprea::ASTBuilder &builder, GazpreaParser::FunctionBlockTupleReturnContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(gazprea::ASTBuilder &builder, GazpreaParser::FunctionPrototypeTupleReturnContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(gazprea::ASTBuilder &builder, GazpreaParser::FunctionStatContext *ctx);


CompleteType ExtractReturnType(gazprea::ASTBuilder &builder, GazpreaParser::FunctionBlockContext *ctx);
CompleteType ExtractReturnType(gazprea::ASTBuilder &builder, GazpreaParser::FunctionPrototypeContext *ctx);
CompleteType ExtractReturnType(gazprea::ASTBuilder &builder, GazpreaParser::FunctionStatContext *ctx);
CompleteType ExtractReturnType(gazprea::ASTBuilder &builder, GazpreaParser::FunctionBlockTupleReturnContext *ctx);
CompleteType ExtractReturnType(gazprea::ASTBuilder &builder, GazpreaParser::FunctionPrototypeTupleReturnContext *ctx);

// Convenience helper: convert ExtractParams output to VarInfo vector
std::vector<VarInfo> ParamsToVarInfo(const std::vector<std::pair<CompleteType, std::string>>& params,
                                      bool isConstDefault = true);


}} // namespaces