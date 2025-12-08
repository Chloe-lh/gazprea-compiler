#include "CompileTimeExceptions.h"
#include "MLIRgen.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"

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

    // Wrap element into VarInfo and push, handling array/vector/matrix fields as descriptors
    CompleteType elemType = structVarInfo->type.subTypes[node->fieldIndex];

    if (elemType.baseType == BaseType::ARRAY ||
        elemType.baseType == BaseType::VECTOR ||
        elemType.baseType == BaseType::MATRIX ||
        elemType.baseType == BaseType::STRING) {
        // For composite array-like fields, treat the extracted value as an LLVM descriptor.
        VarInfo elementVarInfo(elemType);
        elementVarInfo.value = elemVal;      // LLVM struct descriptor {ptr, len}/ {ptr, rows, cols}
        elementVarInfo.isLValue = false;
        // runtimeDims can be inferred lazily by helpers like computeArraySize.
        pushValue(elementVarInfo);
    } else {
        // Scalar / non-array-like field: keep existing behavior.
        VarInfo elementVarInfo(elemType);
        allocaLiteral(&elementVarInfo, node->line);
        builder_.create<mlir::memref::StoreOp>(
            loc_, elemVal, elementVarInfo.value, mlir::ValueRange{}
        );
        pushValue(elementVarInfo);
    }
}

void MLIRGen::visit(StructAccessAssignStatNode* node) {
    if (!node || !node->target || !node->expr) throw std::runtime_error("MLIRGen::StructAccessAssignStatNode: Missing target/expr");

    node->expr->accept(*this);
    VarInfo rhsVarInfo = popValue();

    StructAccessNode* target = node->target.get();
    VarInfo* lhsVarInfo = target->binding;

    if (!lhsVarInfo) throw std::runtime_error("MLIRGen::StructAccessAssignStatNode: null VarInfo for lhs");
    if (lhsVarInfo->type.baseType != BaseType::STRUCT) throw std::runtime_error("MLIRGen::StructAccessAssignStatNode: Non-struct lhs");

    // Get field type of lhs
    CompleteType* fieldType = &lhsVarInfo->type.subTypes[node->target->fieldIndex];

    mlir::Value elemVal;
    if (fieldType->baseType == BaseType::ARRAY ||
        fieldType->baseType == BaseType::VECTOR ||
        fieldType->baseType == BaseType::MATRIX ||
        fieldType->baseType == BaseType::STRING) {
        // For array-like fields, expect rhsVarInfo.value to already be a descriptor
        // with the same LLVM type as the field.
        if (!rhsVarInfo.value) {
            throw std::runtime_error("StructAccessAssignStatNode: rhs has no value for array-like field assignment");
        }
        mlir::Type fieldLLVMTy = getLLVMType(*fieldType);

        if (rhsVarInfo.value.getType() != fieldLLVMTy) {
             // Promote RHS to match LHS field type (this handles int->real, etc. and ensures we have a memref)
             // For STRING, promoteType might return memref (if literal) or descriptor (if var)
             
             VarInfo promoted = promoteType(&rhsVarInfo, fieldType, node->line);
             
             if (promoted.value.getType().isa<mlir::LLVM::LLVMStructType>()) {
                 // Already descriptor
                 elemVal = promoted.value;
             } else if (promoted.value.getType().isa<mlir::LLVM::LLVMPointerType>()) {
                 // Pointer to descriptor (e.g. local variable)
                 elemVal = builder_.create<mlir::LLVM::LoadOp>(loc_, fieldLLVMTy, promoted.value);
             } else if (promoted.value.getType().isa<mlir::MemRefType>()) {
                 mlir::Value memref = promoted.value;
                 
                 // Convert memref to descriptor
                 // 1. Extract pointer
                 mlir::Value ptrAsIdx = builder_.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc_, memref);
                 mlir::Value ptrI64 = builder_.create<mlir::arith::IndexCastOp>(loc_, builder_.getI64Type(), ptrAsIdx);
                 mlir::Value ptr = builder_.create<mlir::LLVM::IntToPtrOp>(loc_, mlir::LLVM::LLVMPointerType::get(&context_), ptrI64);

                 // 2. Create descriptor
                 mlir::Value desc = builder_.create<mlir::LLVM::UndefOp>(loc_, fieldLLVMTy);
                 llvm::SmallVector<int64_t, 1> ptrPos{0};
                 desc = builder_.create<mlir::LLVM::InsertValueOp>(loc_, desc, ptr, ptrPos);

                 // 3. Insert dims
                 int rank = memref.getType().cast<mlir::MemRefType>().getRank();
                 auto i64Ty = builder_.getI64Type();
                 
                 if (rank == 1) {
                     mlir::Value zero = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
                     mlir::Value dim0 = builder_.create<mlir::memref::DimOp>(loc_, memref, zero);
                     mlir::Value dim0I64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, dim0);
                     llvm::SmallVector<int64_t, 1> lenPos{1};
                     desc = builder_.create<mlir::LLVM::InsertValueOp>(loc_, desc, dim0I64, lenPos);
                 } else if (rank == 2) {
                     // matrix case
                     mlir::Value zero = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
                     mlir::Value dim0 = builder_.create<mlir::memref::DimOp>(loc_, memref, zero);
                     mlir::Value dim0I64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, dim0);
                     llvm::SmallVector<int64_t, 1> rowPos{1};
                     desc = builder_.create<mlir::LLVM::InsertValueOp>(loc_, desc, dim0I64, rowPos);

                     mlir::Value one = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
                     mlir::Value dim1 = builder_.create<mlir::memref::DimOp>(loc_, memref, one);
                     mlir::Value dim1I64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, dim1);
                     llvm::SmallVector<int64_t, 1> colPos{2};
                     desc = builder_.create<mlir::LLVM::InsertValueOp>(loc_, desc, dim1I64, colPos);
                 }
                 elemVal = desc;
             } else {
                  throw std::runtime_error("StructAccessAssignStatNode: promoted value is not a memref/descriptor.");
             }
        } else {
             elemVal = rhsVarInfo.value;
        }
    } else {
        // Scalar / non-array-like field: keep existing promotion path.
        if (!rhsVarInfo.value) {
            allocaVar(&rhsVarInfo, node->line);
        }
        VarInfo promoted = promoteType(&rhsVarInfo, fieldType, node->line);
        elemVal = getSSAValue(promoted);
    }

    // load in lhs
    mlir::Type structType = getLLVMType(lhsVarInfo->type);
    mlir::Value llvmStruct = builder_.create<mlir::LLVM::LoadOp>(loc_, structType, lhsVarInfo->value);

    llvm::SmallVector<int64_t, 1> pos{static_cast<int64_t>(node->target->fieldIndex)};
    mlir::Value updatedStruct = builder_.create<mlir::LLVM::InsertValueOp>(loc_, llvmStruct, elemVal, pos);
    builder_.create<mlir::LLVM::StoreOp>(loc_, updatedStruct, lhsVarInfo->value);
}
