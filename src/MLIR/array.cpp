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
    if (arrayVar->type.subTypes.size() != 1) {
        throw std::runtime_error("ArrayAccessAssignStat node: array type must have exactly one element subtype");
    }
    CompleteType elemType = arrayVar->type.subTypes[0];

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

    // Create scalar VarInfo with element's CompleteType.
    // For homogeneous arrays, element type is the single subtype.
    if (arrVarInfo->type.subTypes.size() != 1) {
        throw std::runtime_error("ArrayAccessNode: array type must have exactly one element subtype");
    }
    CompleteType elemType = arrVarInfo->type.subTypes[0];
    VarInfo out(elemType);
    out.value = elemVal;
    out.isLValue = false;
    pushValue(out);
}
void MLIRGen::visit(ArrayTypedDecNode *node) {
    // std::cerr << "[DEUBG] MLIR: visiting ArrayTypedDecNode\n";
    //Resolve declared variable
    VarInfo* declaredVar = currScope_->resolveVar(node->id, node->line);
    if (!declaredVar) {
        throw std::runtime_error("ArrayTypedDec node: variable not declared in scope");
    }

    // Compute total number of elements for static arrays from CompleteType::dims
    size_t totalElems = 1;
    bool allStatic = true;
    if (!declaredVar->type.dims.empty()) {
        for (int d : declaredVar->type.dims) {
            if (d < 0) {
                allStatic = false;
                break;
            }
            totalElems *= static_cast<size_t>(d);
        }
    } else {
        allStatic = false;
    }

    // Allocate storage if not already done
    if (!declaredVar->value) {
        allocaVar(declaredVar, node->line);
    }
    // --- Handle initializer expression ---
    if (node->init) {
        node->init->accept(*this);
        VarInfo literal = popValue();
        // If the initializer is an empty array literal, its type will be
        // an ARRAY with a concrete dim of 0 (or an UNKNOWN element type
        // with dims {0}). In that case there's nothing to assign.
        if (literal.type.baseType == BaseType::ARRAY && literal.type.subTypes[0]==BaseType::EMPTY) {
            return;
        }
        assignTo(&literal, declaredVar, node->line);
        return;
    }

    // --- Zero-initialize array if no initializer and static size known ---
    if (allStatic) {
        size_t n = totalElems;
        if (declaredVar->type.subTypes.size() != 1) {
            throw std::runtime_error("ArrayTypedDecNode: array type must have exactly one element subtype");
        }

        // Zero initialize for each type
        CompleteType elemType = declaredVar->type.subTypes[0];
        mlir::Value zeroVal;
        switch (elemType.baseType) {
            case BaseType::INTEGER: {
                auto t = builder_.getI32Type();
                zeroVal = builder_.create<mlir::arith::ConstantOp>(
                    loc_, t, builder_.getIntegerAttr(t, 0));
                break;
            }
            case BaseType::REAL: {
                auto t = builder_.getF32Type();
                zeroVal = builder_.create<mlir::arith::ConstantOp>(
                    loc_, t, builder_.getFloatAttr(t, 0.0));
                break;
            }
            case BaseType::BOOL: {
                auto t = builder_.getI1Type();
                zeroVal = builder_.create<mlir::arith::ConstantOp>(
                    loc_, t, builder_.getIntegerAttr(t, 0));
                break;
            }
            case BaseType::CHARACTER: {
                auto t = builder_.getI8Type();
                zeroVal = builder_.create<mlir::arith::ConstantOp>(
                    loc_, t, builder_.getIntegerAttr(t, 0));
                break;
            }
            default:
                throw std::runtime_error("ArrayTypedDecNode: unsupported element type for zero initialization");
        }

        mlir::Type idxTy = builder_.getIndexType();
        for (size_t i = 0; i < n; ++i) {
            mlir::Value idx = builder_.create<mlir::arith::ConstantOp>(
                loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(i)));
            builder_.create<mlir::memref::StoreOp>(
                loc_, zeroVal, declaredVar->value, mlir::ValueRange{idx});
        }
    }
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

    // Debug: check what type we have
    // std::cerr << "[DEBUG] ArrayLiteralNode: baseType=" << toString(node->type.baseType) 
    //           << " dims.size()=" << node->type.dims.size();
    // if (!node->type.dims.empty()) {
    //     std::cerr << " dims=[" << node->type.dims[0];
    //     if (node->type.dims.size() > 1) std::cerr << "," << node->type.dims[1];
    //     std::cerr << "]";
    // }
    // std::cerr << std::endl;

    VarInfo arrVarInfo(node->type);
    allocaLiteral(&arrVarInfo, node->line);

    if (!arrVarInfo.value) {
        throw std::runtime_error("ArrayLiteralNode: failed to allocate array storage.");
    }

    // Handle 2D array/matrix literals
    if (node->type.dims.size() == 2) {
        int rows = node->type.dims[0];
        int cols = node->type.dims[1];
        
        for (size_t i = 0; i < node->list->list.size(); ++i) {
            // Each element is itself an ArrayLiteralNode
            auto innerArray = std::dynamic_pointer_cast<ArrayLiteralNode>(node->list->list[i]);
            if (!innerArray) {
                throw std::runtime_error("ArrayLiteralNode: expected nested array for 2D literal");
            }
            
            // Visit inner array elements directly (don't visit the inner ArrayLiteralNode)
            for (size_t j = 0; j < innerArray->list->list.size(); ++j) {
                innerArray->list->list[j]->accept(*this);
                VarInfo elem = popValue();
                mlir::Value val = getSSAValue(elem);
                
                auto rowIdx = builder_.create<mlir::arith::ConstantOp>(
                    loc_, builder_.getIndexType(), 
                    builder_.getIntegerAttr(builder_.getIndexType(), static_cast<int64_t>(i)));
                auto colIdx = builder_.create<mlir::arith::ConstantOp>(
                    loc_, builder_.getIndexType(), 
                    builder_.getIntegerAttr(builder_.getIndexType(), static_cast<int64_t>(j)));
                    
                builder_.create<mlir::memref::StoreOp>(
                    loc_, val, arrVarInfo.value, mlir::ValueRange{rowIdx, colIdx});
            }
        }
    } else {
        // Handle 1D array literals
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
    }
    
    pushValue(arrVarInfo);
}

void MLIRGen::visit(RangeExprNode *node)          { std::cout << "RangeExprNode not implemented\n"; }
