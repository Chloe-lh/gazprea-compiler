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

void MLIRGen::visit(StructAccessNode* node) {
    if (!currScope_) {
        throw std::runtime_error("StructAccessNode: no current scope");
    }

    VarInfo* structVarInfo = node->binding;
    if (!structVarInfo) {
        throw std::runtime_error("StructAccessNode: no bound tuple variable for '" + node->structName + "'");
    }

    if (structVarInfo->type.baseType != BaseType::STRUCT) {
        throw std::runtime_error("TupleAccessNode: Variable '" + node->structName + "' is not a tuple.");
    }

    // Try resolving as global
    if (!structVarInfo->value) {
        auto globalOp = module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->structName);
        if (globalOp) {
            structVarInfo->value = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
        } else {
            allocaVar(structVarInfo, node->line);
        }
    }

    // Should have resolved a value by now
    if (!structVarInfo->value) throw std::runtime_error("MLIRGen::StructAccessNode: Struct variable '" + node->structName + "' has no value.");

    // Extract element by field index
    mlir::Type structTy = getLLVMType(structVarInfo->type);
    mlir::Value structVal = builder_.create<mlir::LLVM::LoadOp>(loc_, structTy, structVarInfo->value);
    llvm::SmallVector<int64_t, 1> pos{static_cast<int64_t>(node->fieldIndex)};
    mlir::Value elemVal = builder_.create<mlir::LLVM::ExtractValueOp>(loc_, structVal, pos);

    // Wrap element into scalar VarInfo and push
    CompleteType elemType = structVarInfo->type.subTypes[node->fieldIndex];
    VarInfo elementVarInfo(elemType);
    allocaLiteral(&elementVarInfo, node->line);
    builder_.create<mlir::LLVM::StoreOp>(loc_, elemVal, elementVarInfo.value);
    pushValue(elementVarInfo);
}
