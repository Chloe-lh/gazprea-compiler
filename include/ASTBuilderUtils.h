#pragma once
#include "Types.h"
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
};

namespace gazprea { class ASTBuilder; }
namespace gazprea { namespace builder_utils {

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(gazprea::ASTBuilder &builder, GazpreaParser::FunctionBlockContext *ctx);

std::vector<std::pair<CompleteType, std::string>>
ExtractParams(gazprea::ASTBuilder &builder, GazpreaParser::ProcedureContext *ctx);

CompleteType ExtractReturnType(gazprea::ASTBuilder &builder, GazpreaParser::FunctionBlockContext *ctx);
CompleteType ExtractReturnType(gazprea::ASTBuilder &builder, GazpreaParser::FunctionStatContext *ctx);


}} // namespaces