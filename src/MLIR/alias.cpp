#include "MLIRgen.h"

/* Resolve aliases using currScope_->resolveAlias */
void MLIRGen::visit(TypeAliasDecNode* node) {
    /* Nothing to do - already declared during semantic analysis. */
}

/* Resolve aliases using currScope_->resolveAlias */
void MLIRGen::visit(TypeAliasNode* node) {
    /* Nothing to do - already declared during semantic analysis. */
}

/* Resolve aliases using currScope_->resolveAlias */
void MLIRGen::visit(TupleTypeAliasNode* node) {
    /* Nothing to do - already declared during semantic analysis. */
}