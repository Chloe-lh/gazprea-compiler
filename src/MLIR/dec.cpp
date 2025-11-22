#include "CompileTimeExceptions.h"
#include "MLIRgen.h"

void MLIRGen::visit(TypedDecNode* node) {
    // Resolve variable declared by semantic analysis
    VarInfo* declaredVar = currScope_->resolveVar(node->name);

    // defensive sync for qualifier flag
    if (node->qualifier == "const") {
        declaredVar->isConst = true;
    } else if (node->qualifier == "var") {
        declaredVar->isConst = false;
    }

    // Ensure storage exists regardless of initializer
    if (!declaredVar->value) {
        allocaVar(declaredVar);
    }

    // Handle optional initializer + promotion
    if (node->init) {
        node->init->accept(*this);
        VarInfo literal = popValue();
        assignTo(&literal, declaredVar);
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
    VarInfo* declaredVar = currScope_->resolveVar(node->name);


    // Semantic analysis should have handled this - this is just in casse
    if (node->qualifier == "const") {
        declaredVar->isConst = true;
    } else if (node->qualifier == "var") {
        declaredVar->isConst = false;
    } else {
        throw StatementError(1, "Cannot infer variable '" + node->name + "' without qualifier."); // TODO: line number
    }
    
    assignTo(&literal, declaredVar);
}