#include "CompileTimeExceptions.h"
#include "MLIRgen.h"

// Statements
void MLIRGen::visit(AssignStatNode* node) {
    if (!node->expr) throw std::runtime_error("FATAL: No expr for assign stat found"); 
    node->expr->accept(*this);
    VarInfo* to = currScope_->resolveVar(node->name);
    VarInfo from = popValue();

    if (to->isConst) {
        throw AssignError(1, "Cannot assign to const variable '" + to->identifier + "'.");
    }

    assignTo(&from, to);
}

void MLIRGen::visit(DestructAssignStatNode* node) {
    if (!node->expr) {
        throw std::runtime_error("FATAL: No expression for destructuring assignment.");
    }

    // Evaluate RHS tuple expression
    node->expr->accept(*this);
    VarInfo fromTuple = popValue();

    if (fromTuple.type.baseType != BaseType::TUPLE) {
        throw AssignError(1, "Codegen: destructuring assignment requires a tuple expression on the right-hand side.");
    }

    if (fromTuple.type.subTypes.size() != node->names.size()) {
        throw AssignError(1, "Codegen: tuple arity mismatch in destructuring assignment.");
    }

    // Destructure RHS tuple into individual target variables.
    // TODO: support element-wise promotions if needed; for now we rely on
    // semantic analysis to ensure exact type matches.
    if (!fromTuple.value) {
        allocaVar(&fromTuple);
    }

    for (size_t i = 0; i < node->names.size(); ++i) {
        const std::string &name = node->names[i];
        VarInfo* target = currScope_->resolveVar(name);
        if (!target) {
            throw SymbolError(1, "Codegen: variable '" + name + "' not defined in destructuring assignment.");
        }
        if (target->isConst) {
            throw AssignError(1, "Codegen: cannot assign to const variable '" + name + "' in destructuring assignment.");
        }

        // Ensure scalar storage exists
        if (target->type.baseType != BaseType::TUPLE && !target->value) {
            allocaVar(target);
        }
        if (target->type.baseType == BaseType::TUPLE) {
            throw AssignError(1, "Codegen: nested tuple destructuring not supported in codegen.");
        }

        // Extract element from tuple struct and store into target
        mlir::Type tupleStructTy = getLLVMType(fromTuple.type);
        mlir::Value tupleStruct = builder_.create<mlir::LLVM::LoadOp>(
            loc_, tupleStructTy, fromTuple.value);
        llvm::SmallVector<int64_t, 1> pos{static_cast<int64_t>(i)};
        mlir::Value elemVal =
            builder_.create<mlir::LLVM::ExtractValueOp>(loc_, tupleStruct, pos);
        builder_.create<mlir::memref::StoreOp>(
            loc_, elemVal, target->value, mlir::ValueRange{});
    }
}