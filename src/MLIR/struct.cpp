#include "CompileTimeExceptions.h"
#include "MLIRgen.h"

void MLIRGen::visit(StructTypedDecNode* node) {
    // StructTypedDecNode behaves like a typed declaration from the
    // codegen point of view: semantic analysis has already declared
    // the variable (if any) with the correct struct type.
    if (node->name.empty()) {
        return; // pure type declaration; no code to emit
    }

    VarInfo* declaredVar = currScope_->resolveVar(node->name, node->line);

    // Ensure storage exists regardless of initializer
    if (!declaredVar->value) {
        allocaVar(declaredVar, node->line);
    }

    if (node->init) {
        node->init->accept(*this);
        VarInfo literal = popValue();
        assignTo(&literal, declaredVar, node->line);
    } else {
        zeroInitializeVar(declaredVar);
    }
}
