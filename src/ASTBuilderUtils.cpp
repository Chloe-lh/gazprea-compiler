#include "ASTBuilderUtils.h"
#include "ASTBuilder.h"
#include "GazpreaParser.h"
#include "Types.h"
#include <any>
#include <cstddef>

using namespace gazprea;
using namespace gazprea::builder_utils;



// Given parameter list, extract each parameter and return in a list
//: FUNCTION ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT RETURNS type block       #FunctionBlock

// 
namespace gazprea { namespace builder_utils { 

    std::vector<std::pair<CompleteType, std::string>>
    ExtractParams(gazprea::ASTBuilder &builder, gazprea::GazpreaParser::FunctionBlockContext *ctx) {
        std::vector<std::pair<CompleteType, std::string>> params;
        if (!ctx) return params;

        size_t idCount = ctx->ID().size();
        size_t paramCount = (idCount > 0) ? idCount - 1 : 0;

        for (size_t i = 0; i < paramCount; ++i) {
            CompleteType ptype(BaseType::UNKNOWN);

            if (ctx->type().size() > i) {
                auto anyT = builder.visit(ctx->type(i)); // dispatches to visitType
                if (anyT.has_value() && anyT.type() == typeid(CompleteType)) {
                    ptype = std::any_cast<CompleteType>(anyT); // unbox the CompleteType
                }
            }
            std::string pname;
            if (ctx->ID(i + 1)) pname = ctx->ID(i + 1)->getText();
            if (pname.empty()) pname = "_arg" + std::to_string(i);

            params.emplace_back(std::move(ptype), std::move(pname));
        }
        return params;
    }
    
    CompleteType ExtractReturnType(ASTBuilder &builder, GazpreaParser::FunctionBlockContext *ctx) {
        if (!ctx) return CompleteType(BaseType::UNKNOWN);

        size_t idCount = ctx->ID().size();
        size_t paramCount = (idCount > 0) ? idCount - 1 : 0;

        if (ctx->type().size() > paramCount) {
            auto anyT = builder.visit(ctx->type(paramCount));
            if (anyT.has_value() && anyT.type() == typeid(CompleteType)) {
                return std::any_cast<CompleteType>(anyT);
            }
            // fallback to textual mapping for primitives
            auto ttext = ctx->type(paramCount)->getText();
            if (ttext == "integer") return CompleteType(BaseType::INTEGER);
            if (ttext == "real") return CompleteType(BaseType::REAL);
            if (ttext == "character") return CompleteType(BaseType::CHARACTER);
            if (ttext == "boolean") return CompleteType(BaseType::BOOL);
        }
        return CompleteType(BaseType::UNKNOWN);
    }
}} // namespace gazprea::builder_utils
