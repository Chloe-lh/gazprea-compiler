#include "AST.h"
#include "CompileTimeExceptions.h"
#include "MLIRgen.h"
#include "Types.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <stdexcept>


void MLIRGen::visit(ArrayStrideExpr *node)        { std::cout << "ArrayStrideExpr not implemented\n"; }
void MLIRGen::visit(ArraySliceExpr *node)         { std::cout << "ArraySliceExpr not implemented\n"; }
void MLIRGen::visit(ArrayAccessAssignStatNode *node) { 
    if (!node->target || !node->expr) {
        throw std::runtime_error("ArrayAccessAssignStat node: missing target or expression");
    }

    // Evaluate RHS
    node->expr->accept(*this);
    VarInfo from = popValue();
    if (from.type.baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("ArrayAccessAssignStat node: RHS side has UNKNOWN type");
    }

    ArrayAccessNode* target = node->target.get();
    VarInfo* arrayVar = target->binding;
    if (!arrayVar) throw std::runtime_error("ArrayAccessAssignStat node: target has no bound array");
    if (!arrayVar->value) allocaVar(arrayVar, node->line);
    if (!arrayVar->value) throw std::runtime_error("ArrayAccessAssignStat node: array has no storage");

    // convert 1-based -> 0-based
    if (target->index <= 0) throw std::runtime_error("ArrayAccessAssignStat: invalid index (must be >= 1)");
    size_t idx0 = static_cast<size_t>(target->index - 1);

    // --- element type ---
    CompleteType elemType = arrayVar->type.elemType;  // use elemType directly

    // Promote RHS to element type
    VarInfo promoted = promoteType(&from, &elemType, node->line);
    mlir::Value val = getSSAValue(promoted);

    // Build index and store
    auto idxTy = builder_.getIndexType();
    auto idxConst = builder_.create<mlir::arith::ConstantOp>(
        loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(idx0))
    );

    builder_.create<mlir::memref::StoreOp>(
        loc_, val, arrayVar->value, mlir::ValueRange{idxConst}
    );
}

void MLIRGen::visit(ArrayAccessNode *node) {
    if (!currScope_) throw std::runtime_error("ArrayAccessNode: no current scope");
    VarInfo* arrVarInfo = node->binding;
    if (!arrVarInfo) throw std::runtime_error("ArrayAccessNode: unresolved array '"+node->id+"'");
    if (arrVarInfo->type.baseType != BaseType::ARRAY)
        throw std::runtime_error("ArrayAccessNode: Variable '" + node->id + "' is not an array.");

    if (!arrVarInfo->value) {
        auto globalOp = module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->id);
        if (globalOp) {
            arrVarInfo->value = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
        } else {
            allocaVar(arrVarInfo, node->line);
        }
    }
    if (!arrVarInfo->value) throw std::runtime_error("ArrayAccessNode: array has no storage");

    if (node->index <= 0) throw std::runtime_error("ArrayAccessNode: invalid index (must be >= 1)");
    size_t idx0 = static_cast<size_t>(node->index - 1);
    // compute index value (1-based)
    auto idxTy = builder_.getIndexType();
    auto idxConst = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(idx0)));

    // memref.load element
    mlir::Value elemVal = builder_.create<mlir::memref::LoadOp>(loc_, arrVarInfo->value, mlir::ValueRange{idxConst});

    // Create scalar VarInfo with element's CompleteType
    CompleteType elemType = !arrVarInfo->type.subTypes.empty()
                             ? arrVarInfo->type.subTypes[idx0]
                             : CompleteType(arrVarInfo->type.elemType);
    VarInfo out(elemType);
    out.value = elemVal;
    out.isLValue = false;
    pushValue(out);
}
// ArrayTypedDec node has a ArrayType node with size info and resolved Dims
void MLIRGen::visit(ArrayTypedDecNode *node) {
    // std::cerr << "[DEUBG] MLIR: visiting ArrayTypedDecNode\n";
    //Resolve declared variable
    VarInfo* declaredVar = currScope_->resolveVar(node->id, node->line);
    if (!declaredVar) {
        throw std::runtime_error("ArrayTypedDec node: variable not declared in scope");
    }

    if (!node->typeInfo) {
        throw std::runtime_error("ArrayTypedDec node: missing typeInfo");
    }

    // Element type (already resolved by semantic analysis)
    CompleteType elemCT = node->typeInfo->elementType;

    // Compute total number of elements for static arrays
    size_t totalElems = 1;
    bool allStatic = true;
    for (auto &dim : node->typeInfo->resolvedDims) {
        if (!dim.has_value()) { 
            allStatic = false; 
            break; 
        }
        if (dim.value() < 0) throw std::runtime_error("Negative array size");
        totalElems *= static_cast<size_t>(dim.value());
    }

    // Preserve/compute array size information for codegen (total elems)
    if (allStatic && !node->typeInfo->resolvedDims.empty()) {
        declaredVar->arraySize = totalElems; // store total elements for bounds & zero-init
    } else {
        declaredVar->arraySize.reset(); // dynamic array
    }

    // Allocate storage if not already done
    if (!declaredVar->value) {
        allocaVar(declaredVar, node->line);
    }
    // --- Handle initializer expression ---
    if (node->init) {
        node->init->accept(*this);
        VarInfo literal = popValue();
        // std::cerr << "[DEBUG:ASSIGN] Declared type: " << toString(declaredVar->type)
        //   << ", initializer type: " << toString(literal.type)
        //   << ", line: " << node->line << "\n";
        assignTo(&literal, declaredVar, node->line);
        return;
    }

    // --- Zero-initialize array if no initializer ---
    if (declaredVar->arraySize.has_value()) {
        size_t n = declaredVar->arraySize.value();
        mlir::Type idxTy = builder_.getIndexType();
        for (size_t i = 0; i < n; ++i) {
            mlir::Value idx = builder_.create<mlir::arith::ConstantOp>(
                loc_, idxTy, builder_.getIntegerAttr(idxTy, i)
            );
            mlir::Value elemPtr = builder_.create<mlir::memref::LoadOp>(
                loc_, declaredVar->value, idx
            );
            VarInfo elemVar(declaredVar->type.elemType);
            elemVar.value = elemPtr;
            zeroInitializeVar(&elemVar);
        }
    }
}

// //holds elementType, sizeExprs, resolvedDims
void MLIRGen::visit(ArrayTypeNode *node){ 
    // if(!node.sizeExprs.empty()){
    //     for(const auto &sz: node.sizeExprs){
            
    //     }
    // }
    // map MLIR type to MLIR scalar types
    // if vector, shape dynamic vector
    // for each dimension, push a value or a dynamic shape -> use sizeExprs
    //convert lowered SSA into index type
    
    // bool = i1
    // float/real/int  = i64


}
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
        // std::cerr << "[DEUBG] MLIR: visiting ArrayLiteralNode\n";

    VarInfo arrVarInfo(node->type);
    // If the literal has a known element count, allocate a fixed-size
    // memref so allocaVar can create a static AllocaOp. Otherwise the
    // dynamic-allocation path is not yet implemented in allocaVar.
    if (node->list && node->list->list.size() > 0) {
        arrVarInfo.arraySize = static_cast<int64_t>(node->list->list.size());
    }
    allocaLiteral(&arrVarInfo, node->line);

    if (!arrVarInfo.value) {
        throw std::runtime_error("ArrayLiteralNode: failed to allocate array storage.");
    }

    std::cout << (node->list->list.size());
    for (size_t i = 0; i < node->list->list.size(); ++i) {
        node->list->list[i]->accept(*this);
        VarInfo elem = popValue();
        mlir::Value val = getSSAValue(elem);
        // MLIR memref uses 0-based index
        auto idxConst = builder_.create<mlir::arith::ConstantOp>(
            loc_, builder_.getIndexType(), builder_.getIntegerAttr(builder_.getIndexType(), static_cast<int64_t>(i))
        );
        builder_.create<mlir::memref::StoreOp>(loc_, val, arrVarInfo.value, mlir::ValueRange{idxConst});
    }
    pushValue(arrVarInfo);
}

void MLIRGen::visit(RangeExprNode *node)          { std::cout << "RangeExprNode not implemented\n"; }