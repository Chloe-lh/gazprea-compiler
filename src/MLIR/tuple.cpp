#include "MLIRgen.h"

void MLIRGen::visit(TupleTypedDecNode* node) {
    // Resolve variable declared by semantic analysis
    VarInfo* declaredVar = currScope_->resolveVar(node->name, node->line);

    // Ensure storage for tuple value exists
    if (!declaredVar->value) {
        allocaVar(declaredVar, node->line);
    }

    // Handle optional initializer
    if (node->init) {
        node->init->accept(*this);
        VarInfo literal = popValue();
        assignTo(&literal, declaredVar, node->line);
    }
}

void MLIRGen::visit(TupleAccessAssignStatNode* node) {
    if (!node->target || !node->expr) {
        throw std::runtime_error("TupleAccessAssignStatNode: missing target or expression");
    }

    // Evaluate RHS expression
    node->expr->accept(*this);
    VarInfo from = popValue();
    if (from.type.baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("TupleAccessAssignStatNode: RHS has UNKNOWN type");
    }

    TupleAccessNode* target = node->target.get();
    VarInfo* tupleVar = target->binding;
    if (!tupleVar) {
        throw std::runtime_error("TupleAccessAssignStatNode: target has no bound tuple variable");
    }
    if (tupleVar->type.baseType != BaseType::TUPLE) {
        throw std::runtime_error("TupleAccessAssignStatNode: target variable '" +
                                 target->tupleName + "' is not a tuple");
    }

    // Ensure tuple storage exists
    if (!tupleVar->value) {
        allocaVar(tupleVar, node->line);
    }

    if (target->index < 1 || static_cast<size_t>(target->index) > tupleVar->type.subTypes.size()) {
        throw std::runtime_error("TupleAccessAssignStatNode: index out of range for tuple '" +
                                 target->tupleName + "'");
    }

    // Compute element type (1-based index in AST)
    size_t elemIndex = static_cast<size_t>(target->index - 1);
    CompleteType elemType = tupleVar->type.subTypes[elemIndex];

    // Promote RHS to element type if needed and normalize to SSA
    VarInfo promoted = promoteType(&from, &elemType, node->line);
    mlir::Value elemVal = getSSAValue(promoted);

    // Load current tuple struct from LLVM pointer
    mlir::Type tupleStructTy = getLLVMType(tupleVar->type);
    mlir::Value tupleStruct = builder_.create<mlir::LLVM::LoadOp>(
        loc_, tupleStructTy, tupleVar->value);

    // Insert updated element and store back
    llvm::SmallVector<int64_t, 1> pos{static_cast<int64_t>(elemIndex)};
    mlir::Value updatedStruct =
        builder_.create<mlir::LLVM::InsertValueOp>(loc_, tupleStruct, elemVal, pos);
    builder_.create<mlir::LLVM::StoreOp>(
        loc_, updatedStruct, tupleVar->value);
}

void MLIRGen::visit(TupleAccessNode* node) {
    if (!currScope_) {
        throw std::runtime_error("TupleAccessNode: no current scope");
    }

    VarInfo* tupleVarInfo = node->binding;
    if (!tupleVarInfo) {
        throw std::runtime_error("TupleAccessNode: no bound tuple variable for '" + node->tupleName + "'");
    }

    if (tupleVarInfo->type.baseType != BaseType::TUPLE) {
        throw std::runtime_error("TupleAccessNode: Variable '" + node->tupleName + "' is not a tuple.");
    }

    // Handle tuple as global if found in global lookup
    if (!tupleVarInfo->value) {
        auto globalOp =
            module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->tupleName);
        if (globalOp) {
            tupleVarInfo->value =
                builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
        } else {
            // Local but unallocated tuple
            allocaVar(tupleVarInfo, node->line);
        }
    }

    if (!tupleVarInfo->value) {
        throw std::runtime_error("TupleAccessNode: Tuple variable '" + node->tupleName + "' has no allocated value.");
    }

    // Validate index (1-based)
    if (node->index < 1 ||
        node->index > static_cast<int>(tupleVarInfo->type.subTypes.size())) {
        throw std::runtime_error("TupleAccessNode: Index " +
                                 std::to_string(node->index) +
                                 " out of range for tuple of size " +
                                 std::to_string(tupleVarInfo->type.subTypes.size()));
    }

    // Extract the element at the specified index (convert from 1-based to 0-based)
    mlir::Type structTy = getLLVMType(tupleVarInfo->type);
    mlir::Value structVal = builder_.create<mlir::LLVM::LoadOp>(
        loc_, structTy, tupleVarInfo->value);
    llvm::SmallVector<int64_t, 1> pos{
        static_cast<int64_t>(node->index - 1)};
    mlir::Value elemVal =
        builder_.create<mlir::LLVM::ExtractValueOp>(loc_, structVal, pos);

    // Wrap element into a scalar VarInfo and push it.
    CompleteType elemType =
        tupleVarInfo->type.subTypes[static_cast<size_t>(node->index - 1)];
    VarInfo elementVarInfo(elemType);
    allocaLiteral(&elementVarInfo, node->line);
    builder_.create<mlir::memref::StoreOp>(
        loc_, elemVal, elementVarInfo.value, mlir::ValueRange{});

    pushValue(elementVarInfo);
}

void MLIRGen::visit(TupleTypeCastNode* node) {
    if (!node || !node->expr) {
        throw std::runtime_error("TupleTypeCastNode: missing expression");
    }

    // Evaluate the source tuple expression and delegate to castType.
    node->expr->accept(*this);
    VarInfo sourceTuple = popValue();
    VarInfo resultTuple = castType(&sourceTuple, &node->targetTupleType, node->line);
    pushValue(resultTuple);
}

// could be moved to tuple
void MLIRGen::visit(TupleLiteralNode* node) {
    VarInfo tupleVarInfo(node->type);
    allocaLiteral(&tupleVarInfo, node->line);

    if (!tupleVarInfo.value) {
        throw std::runtime_error("TupleLiteralNode: failed to allocate tuple storage.");
    }
    if (node->type.subTypes.size() != node->elements.size()) {
        throw std::runtime_error(
            "FATAL: mismatched tuple type and element count in literal.");
    }

    mlir::Type structTy = getLLVMType(node->type);
    mlir::Value structVal =
        builder_.create<mlir::LLVM::UndefOp>(loc_, structTy);

    for (size_t i = 0; i < node->elements.size(); ++i) {
        node->elements[i]->accept(*this);
        VarInfo elemVarInfo = popValue();

        // Normalize element value to SSA (load memref if needed)
        mlir::Value loadedVal = getSSAValue(elemVarInfo);

        llvm::SmallVector<int64_t, 1> pos{static_cast<int64_t>(i)};
        structVal = builder_.create<mlir::LLVM::InsertValueOp>(
            loc_, structVal, loadedVal, pos);
    }

    builder_.create<mlir::LLVM::StoreOp>(
        loc_, structVal, tupleVarInfo.value);

    pushValue(tupleVarInfo);
}
