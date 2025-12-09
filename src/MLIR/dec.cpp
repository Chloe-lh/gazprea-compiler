#include "CompileTimeExceptions.h"
#include "MLIRgen.h"

void MLIRGen::visit(TypedDecNode* node) {
    // Resolve variable declared by semantic analysis
    VarInfo* declaredVar = currScope_->resolveVar(node->name, node->line);

    // defensive sync for qualifier flag
    if (node->qualifier == "const") {
        declaredVar->isConst = true;
    } else if (node->qualifier == "var") {
        declaredVar->isConst = false;
    }

    // Ensure storage exists regardless of initializer
    if (!declaredVar->value) {
        allocaVar(declaredVar, node->line);
    }

    // Handle optional initializer + promotion
    if (node->init) {
        node->init->accept(*this);
        VarInfo literal = popValue();
        assignTo(&literal, declaredVar, node->line);
    } else {
        // Implicit zero-initialization for scalars
        zeroInitializeVar(declaredVar);
    }
}

/* Functionally the same as TypedDecNode except initializer is required */
void MLIRGen::visit(InferredDecNode* node) {
    if (!node->init) {
        throw std::runtime_error("FATAL: Inferred declaration without initializer.");
    }
    node->init->accept(*this); // Resolve init value

    VarInfo literal = popValue();
    VarInfo* declaredVar = currScope_->resolveVar(node->name, node->line);


    // Semantic analysis should have handled this - this is just in casse
    if (node->qualifier == "const") {
        declaredVar->isConst = true;
    } else if (node->qualifier == "var") {
        declaredVar->isConst = false;
    } else {
        throw StatementError(node->line, "Cannot infer variable '" + node->name + "' without qualifier."); // TODO: line number
    }

    // Ensure storage exists for dynamic arrays (needs runtime size) before assignment
    if (declaredVar->type.baseType == BaseType::ARRAY &&
        declaredVar->type.dims.size() == 1 &&
        declaredVar->type.dims[0] < 0 &&
        !declaredVar->value) {
        mlir::Value sizeValue = computeArraySize(&literal, node->line);
        allocaVar(declaredVar, node->line, sizeValue);
    } else if (!declaredVar->value) {
        allocaVar(declaredVar, node->line);
    }

    assignTo(&literal, declaredVar, node->line);
}