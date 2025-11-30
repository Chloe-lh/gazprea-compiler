#include "ASTBuilder.h"
#include "ASTBuilderUtils.h"
#include "AST.h"

namespace gazprea {
    std::any
    ASTBuilder::visitStructTypedDec(GazpreaParser::StructTypedDecContext *ctx) {
    std::string qualifier = "const";

    // 
    if (ctx->qualifier()) {
        auto qualAny = visit(ctx->qualifier());
        if (qualAny.has_value()) {
        try {
            qualifier = std::any_cast<std::string>(qualAny);
        } catch (const std::bad_any_cast &) {
            throw std::runtime_error("ASTBuilder::visitStructTypedDec(): Failed to cast qualifier");
        }
        }
    }

    // Resolve parser type -> AST type system
    CompleteType structType = CompleteType(BaseType::UNKNOWN);
    auto anyType = visit(ctx->struct_dec());
    structType = std::any_cast<CompleteType>(anyType);

    // If there is a variable declaration, parse its ID. Otherwise leave
    // the variable name empty; only the struct type is being declared.
    // Grammar: `struct_dec (ID (EQ expr)?)?`
    std::string id;
    if (ctx->ID()) {
        id = ctx->ID()->getText();
    }

    // optional initializer expression
    std::shared_ptr<ExprNode> init = nullptr;
    if (ctx->expr()) {
        if (id == "") throw std::runtime_error("ASTBuilder::visitStructTypedDec: initializer provided but no variable identifier provided.");
        auto anyInit = visit(ctx->expr());
        if (anyInit.has_value()) {
        init = safe_any_cast_ptr<ExprNode>(anyInit);
        }
    }

    auto node = std::make_shared<StructTypedDecNode>(id, qualifier, structType);
    setLocationFromCtx(node, ctx);
    node->init = init;
    return node_any(std::move(node));
    }

    std::any ASTBuilder::visitStruct_dec(GazpreaParser::Struct_decContext *ctx) {
    std::vector<CompleteType> elemTypes;
    for (auto typeCtx : ctx->type()) {
        auto anyType = visit(typeCtx);
        elemTypes.push_back(std::any_cast<CompleteType>(anyType));
    }

    // Capture the struct's declared name so semantic analysis can
    // register a named struct type (e.g., for use in `var Name x;`).
    std::string structName;
    if (!ctx->ID().empty()) {
        structName = ctx->ID(0)->getText();
    }
    CompleteType structType(BaseType::STRUCT, std::move(elemTypes));
    structType.aliasName = structName;
    return structType;
    } 
}
