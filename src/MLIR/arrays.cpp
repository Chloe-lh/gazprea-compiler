#include "CompileTimeExceptions.h"
#include "MLIRgen.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include <stdexcept>


void MLIRGen::visit(ArrayStrideExpr *node)        { std::cout << "ArrayStrideExpr not implemented\n"; }
void MLIRGen::visit(ArraySliceExpr *node)         { std::cout << "ArraySliceExpr not implemented\n"; }
void MLIRGen::visit(ArrayAccessExpr *node){ 
    if(!currScope_){
        throw std::runtime_error("ArrayAccessNode: no current scope");
    }
    VarInfo* arrVarInfo = node->binding;
    if(!arrVarInfo){
        throw std::runtime_error("ArrayAccessNode: no bound tuple variable for "+node->id);
    }
    if (arrVarInfo->type.baseType != BaseType::ARRAY) {
        throw std::runtime_error("TupleAccessNode: Variable '" + node->id + "' is not an array.");
    }
    // global array
    if(!arrVarInfo->value){
        auto globalOp = module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->id);
        if(globalOp){
            arrVarInfo->value = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
        }else{
            allocaVar(arrVarInfo, node->line);
        }
    }
    if(!arrVarInfo->value){
        throw std::runtime_error("TupleAccessNode: Tuple variable '" + node->id + "' has no allocated value.");
    }
    // Validate index (1-based)
    if (node->index < 1 ||
        node->index > static_cast<int>(arrVarInfo->type.subTypes.size())) {
        throw std::runtime_error("TupleAccessNode: Index " +
                                 std::to_string(node->index) +
                                 " out of range for tuple of size " +
                                 std::to_string(arrVarInfo->type.subTypes.size()));
    }
        // Extract the element at the specified index (convert from 1-based to 0-based)
    mlir::Type structTy = getLLVMType(arrVarInfo->type);
    mlir::Value structVal = builder_.create<mlir::LLVM::LoadOp>(
        loc_, structTy, arrVarInfo->value);
    llvm::SmallVector<int64_t, 1> pos{
        static_cast<int64_t>(node->index - 1)};
    mlir::Value elemVal =
        builder_.create<mlir::LLVM::ExtractValueOp>(loc_, structVal, pos);

       // Wrap element into a scalar VarInfo and push it.
    CompleteType elemType =
        arrVarInfo->type.subTypes[static_cast<size_t>(node->index - 1)];
    VarInfo elementVarInfo(elemType);
    allocaLiteral(&elementVarInfo, node->line);
    builder_.create<mlir::memref::StoreOp>(
        loc_, elemVal, elementVarInfo.value, mlir::ValueRange{});

    pushValue(elementVarInfo);
}
void MLIRGen::visit(ArrayTypedDecNode *node){ 
    VarInfo* declaredVar = currScope_->resolveVar(node->id, node->line);
    if(!declaredVar->value){
        allocaVar(declaredVar, node->line);
    }
    if(node->typeInfo){
        node->typeInfo->accept(*this);
        VarInfo typeInfo = popValue();
    }
    if(node->init){
        node->init->accept(*this);
        VarInfo literal = popValue();
        assignTo(&literal, declaredVar, node->line);
    }else{// implicit zero initialization
        mlir::Value arrPtr = declaredVar->value;
        mlir::Type structTy = getLLVMType(declaredVar->type);
        mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
        auto i32Ty = builder_.getI32Type();
        mlir::Value zeroIdx = builder_.create<mlir::arith::ConstantOp>(loc_, i32Ty, builder_.getIntegerAttr(i32Ty, 0));
        for (size_t i = 0; i < declaredVar->type.subTypes.size(); ++i) {
            mlir::Value fieldIdx = builder_.create<mlir::arith::ConstantOp>(loc_, i32Ty, builder_.getIntegerAttr(i32Ty, i));
            
            // GEP to get address of the element
            mlir::Value elemPtr = builder_.create<mlir::LLVM::GEPOp>(
                loc_, ptrTy, structTy, arrPtr, mlir::ValueRange{zeroIdx, fieldIdx}
            );
            // Use helper to store zero into this address
            VarInfo elemVar(declaredVar->type.subTypes[i]);
            elemVar.value = elemPtr;
            zeroInitializeVar(&elemVar);
        }
    }
}
void MLIRGen::visit(ArrayTypeNode *node)          { std::cout << "ArrayTypeNode not implemented\n"; }
void MLIRGen::visit(ExprListNode *node){
    // Evaluate each element expression and push its VarInfo onto the value stack.
    // This leaves the element VarInfos on `v_stack_` for callers to pop in order.
    if (!node) return;
    for (auto &elem : node->list) {
        if (elem) {
            elem->accept(*this);
        } else {
            // Push a placeholder UNKNOWN value for null entries.
            VarInfo placeholder{CompleteType(BaseType::UNKNOWN)};
            allocaLiteral(&placeholder, 1);
            pushValue(placeholder);
        }
    }
}
void MLIRGen::visit(ArrayLiteralNode *node){
    // Allocate storage for the array literal and populate it element-wise.
    VarInfo arrVarInfo(node->type);
    allocaLiteral(&arrVarInfo, node->line);

    if (!arrVarInfo.value) {
        throw std::runtime_error("ArrayLiteralNode: failed to allocate array storage.");
    }

    // Determine number of elements in the literal
    size_t nElems = 0;
    if (node->list) nElems = node->list->list.size();

    // Build an LLVM aggregate (undef) and insert elements
    mlir::Type arrTy = getLLVMType(node->type);
    mlir::Value agg = builder_.create<mlir::LLVM::UndefOp>(loc_, arrTy);

    for (size_t i = 0; i < nElems; ++i) {
        auto &elemExpr = node->list->list[i];
        if (!elemExpr) {
            // Insert a zero-initialized element for null entries
            VarInfo zeroElem{ node->type.subTypes.empty() ? CompleteType(BaseType::UNKNOWN) : node->type.subTypes[0] };
            allocaLiteral(&zeroElem, node->line);
            mlir::Value loaded = getSSAValue(zeroElem);
            llvm::SmallVector<int64_t, 1> pos{static_cast<int64_t>(i)};
            agg = builder_.create<mlir::LLVM::InsertValueOp>(loc_, agg, loaded, pos);
            continue;
        }

        // Evaluate element expression and obtain its VarInfo
        elemExpr->accept(*this);
        VarInfo elemVar = popValue();

        // Normalize to SSA value and insert into aggregate
        mlir::Value loadedVal = getSSAValue(elemVar);
        llvm::SmallVector<int64_t, 1> pos{static_cast<int64_t>(i)};
        agg = builder_.create<mlir::LLVM::InsertValueOp>(loc_, agg, loadedVal, pos);
    }

    // Store the aggregate into the allocated literal storage and push the array VarInfo
    builder_.create<mlir::LLVM::StoreOp>(loc_, agg, arrVarInfo.value);
    pushValue(arrVarInfo);
}
void MLIRGen::visit(RangeExprNode *node)          { std::cout << "RangeExprNode not implemented\n"; }