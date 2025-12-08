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

void MLIRGen::visit(ArraySliceExpr *node) {
    if (!currScope_) {
        throw std::runtime_error("ArraySliceExpr: no current scope");
    }

    // Resolve source variable (array or matrix)
    VarInfo* arrVarInfo = currScope_->resolveVar(node->id, node->line);
    if (!arrVarInfo) {
        throw std::runtime_error("ArraySliceExpr: unresolved array '" + node->id + "'");
    }
    if (arrVarInfo->type.baseType == BaseType::MATRIX) {
        throw std::runtime_error("ArraySliceExpr: matrix slicing not yet supported in MLIR.");
    }
    if (arrVarInfo->type.baseType != BaseType::ARRAY) {
        throw std::runtime_error("ArraySliceExpr: Variable '" + node->id + "' is not an array.");
    }

    // Ensure storage exists
    if (!arrVarInfo->value) {
        auto globalOp = module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->id);
        if (globalOp) {
            arrVarInfo->value = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
        } else {
            allocaVar(arrVarInfo, node->line);
        }
    }
    if (!arrVarInfo->value) {
        throw std::runtime_error("ArraySliceExpr: array has no storage");
    }

    auto indexTy = builder_.getIndexType();
    auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    mlir::Value baseLen = builder_.create<mlir::memref::DimOp>(loc_, arrVarInfo->value, zeroIdx);

    // Compute start_0based
    mlir::Value start_0based;
    if (node->range && node->range->start) {
        node->range->start->accept(*this);
        VarInfo startVarInfo = popValue();
        mlir::Value startVal = getSSAValue(startVarInfo);
        mlir::Value startIndex = builder_.create<mlir::arith::IndexCastOp>(loc_, indexTy, startVal);
        auto zeroIndex = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        auto isNegative = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::slt, startIndex, zeroIndex);
        auto oneIndex = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
        auto ifOpStart = builder_.create<mlir::scf::IfOp>(loc_, indexTy, isNegative, true);
        builder_.setInsertionPointToStart(&ifOpStart.getThenRegion().front());
        mlir::Value start_1based_neg = builder_.create<mlir::arith::AddIOp>(loc_, baseLen, startIndex);
        start_1based_neg = builder_.create<mlir::arith::AddIOp>(loc_, start_1based_neg, oneIndex);
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{start_1based_neg});
        builder_.setInsertionPointToStart(&ifOpStart.getElseRegion().front());
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{startIndex});
        builder_.setInsertionPointAfter(ifOpStart);
        mlir::Value start_1based = ifOpStart.getResult(0);
        start_0based = builder_.create<mlir::arith::SubIOp>(loc_, start_1based, oneIndex);
    } else {
        start_0based = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    }

    // Compute end_0based
    mlir::Value end_0based;
    if (node->range && node->range->end) {
        node->range->end->accept(*this);
        VarInfo endVarInfo = popValue();
        mlir::Value endVal = getSSAValue(endVarInfo);
        mlir::Value endIndex = builder_.create<mlir::arith::IndexCastOp>(loc_, indexTy, endVal);
        auto zeroIndex = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        auto isNegative = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::slt, endIndex, zeroIndex);
        auto ifOpEnd = builder_.create<mlir::scf::IfOp>(loc_, indexTy, isNegative, true);
        builder_.setInsertionPointToStart(&ifOpEnd.getThenRegion().front());
        auto oneIndex = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
        mlir::Value end_1based_neg = builder_.create<mlir::arith::AddIOp>(loc_, baseLen, endIndex);
        end_1based_neg = builder_.create<mlir::arith::AddIOp>(loc_, end_1based_neg, oneIndex);
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{end_1based_neg});
        builder_.setInsertionPointToStart(&ifOpEnd.getElseRegion().front());
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{endIndex});
        builder_.setInsertionPointAfter(ifOpEnd);
        mlir::Value end_1based = ifOpEnd.getResult(0);
        auto oneIndex2 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
        end_0based = builder_.create<mlir::arith::SubIOp>(loc_, end_1based, oneIndex2);
    } else {
        end_0based = baseLen;
    }

    // len = end - start + 1
    auto oneIndex = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    auto lenMinusOne = builder_.create<mlir::arith::SubIOp>(loc_, end_0based, start_0based);
    mlir::Value len = builder_.create<mlir::arith::AddIOp>(loc_, lenMinusOne, oneIndex);

    // Array slice path: use runtime helper
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
    mlir::Value basePtr;
    if (arrVarInfo->value.getType().isa<mlir::MemRefType>()) {
        mlir::Value baseIndex = builder_.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc_, arrVarInfo->value);
        auto i64Ty = builder_.getI64Type();
        mlir::Value ptrInt = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, baseIndex);
        basePtr = builder_.create<mlir::LLVM::IntToPtrOp>(loc_, ptrTy, ptrInt);
    } else {
        throw std::runtime_error("ArraySliceExpr: array value is not a memref");
    }

    auto i64Ty = builder_.getI64Type();
    mlir::Value baseLen_i64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, baseLen);
    mlir::Value start_i64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, start_0based);
    mlir::Value end_i64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, end_0based);

    auto sliceFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("gaz_slice_int_checked");
    if (!sliceFunc) {
        throw std::runtime_error("ArraySliceExpr: gaz_slice_int_checked function not found");
    }
    auto callOp = builder_.create<mlir::LLVM::CallOp>(loc_, sliceFunc, mlir::ValueRange{basePtr, baseLen_i64, start_i64, end_i64});
    mlir::Value sliceStruct = callOp.getResult();

    CompleteType sliceType = arrVarInfo->type;
    sliceType.dims.clear();
    sliceType.dims.push_back(-1);
    VarInfo sliceVarInfo(sliceType);
    sliceVarInfo.value = sliceStruct;
    sliceVarInfo.isLValue = false;
    pushValue(sliceVarInfo);
}
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

    // --- element type ---
    if (arrayVar->type.subTypes.size() != 1) {
        throw std::runtime_error("ArrayAccessAssignStat node: array type must have exactly one element subtype");
    }
    CompleteType elemType = arrayVar->type.subTypes[0];

    // Promote RHS to element type
    VarInfo promoted = promoteType(&from, &elemType, node->line);
    mlir::Value val = getSSAValue(promoted);

    // Visit first index expression
    if (!target->indexExpr) {
        throw std::runtime_error("ArrayAccessAssignStat node: missing index expression");
    }
    target->indexExpr->accept(*this);
    VarInfo indexVarInfo = popValue();
    mlir::Value indexVal = getSSAValue(indexVarInfo);

    // Convert to index type and adjust for 0-based indexing
    auto idxTy = builder_.getIndexType();
    mlir::Value indexValAsIndex = builder_.create<mlir::arith::IndexCastOp>(loc_, idxTy, indexVal);
    auto one = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    mlir::Value zeroBasedIndex = builder_.create<mlir::arith::SubIOp>(loc_, indexValAsIndex, one);

    // Check if this is a 2D access
    if (target->indexExpr2) {
        // Visit second index expression for 2D arrays
        target->indexExpr2->accept(*this);
        VarInfo indexVarInfo2 = popValue();
        mlir::Value indexVal2 = getSSAValue(indexVarInfo2);
        
        mlir::Value indexValAsIndex2 = builder_.create<mlir::arith::IndexCastOp>(loc_, idxTy, indexVal2);
        mlir::Value zeroBasedIndex2 = builder_.create<mlir::arith::SubIOp>(loc_, indexValAsIndex2, one);
        
        // Store with two indices for 2D array
        builder_.create<mlir::memref::StoreOp>(
            loc_, val, arrayVar->value, mlir::ValueRange{zeroBasedIndex, zeroBasedIndex2}
        );
    } else {
        // Store with single index for 1D array
        builder_.create<mlir::memref::StoreOp>(
            loc_, val, arrayVar->value, mlir::ValueRange{zeroBasedIndex}
        );
    }
}

void MLIRGen::visit(ArrayAccessNode *node) {
    if (!currScope_) throw std::runtime_error("ArrayAccessNode: no current scope");
    VarInfo* arrVarInfo = node->binding;
    if (!arrVarInfo) throw std::runtime_error("ArrayAccessNode: unresolved array '"+node->id+"'");
    if (arrVarInfo->type.baseType != BaseType::ARRAY && arrVarInfo->type.baseType != BaseType::VECTOR && arrVarInfo->type.baseType != BaseType::MATRIX)
        throw std::runtime_error("ArrayAccessNode: Variable '" + node->id + "' is not an array, matrix, or vector.");

    if (!arrVarInfo->value) {
        auto globalOp = module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->id);
        if (globalOp) {
            arrVarInfo->value = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
        } else {
            allocaVar(arrVarInfo, node->line);
        }
    }
    if (!arrVarInfo->value) throw std::runtime_error("ArrayAccessNode: array has no storage");

    // Visit first index expression
    if (!node->indexExpr) {
        throw std::runtime_error("ArrayAccessNode: missing index expression");
    }
    node->indexExpr->accept(*this);
    VarInfo indexVarInfo = popValue();
    mlir::Value indexVal = getSSAValue(indexVarInfo);

    // Convert to index type and adjust for 0-based indexing
    auto idxTy = builder_.getIndexType();
    mlir::Value indexValAsIndex = builder_.create<mlir::arith::IndexCastOp>(loc_, idxTy, indexVal);
    auto one = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    mlir::Value zeroBasedIndex = builder_.create<mlir::arith::SubIOp>(loc_, indexValAsIndex, one);

    mlir::Value elemVal;
    
    // Check if this is a 2D access
    if (node->indexExpr2) {
        // Visit second index expression for 2D arrays
        node->indexExpr2->accept(*this);
        VarInfo indexVarInfo2 = popValue();
        mlir::Value indexVal2 = getSSAValue(indexVarInfo2);
        
        mlir::Value indexValAsIndex2 = builder_.create<mlir::arith::IndexCastOp>(loc_, idxTy, indexVal2);
        mlir::Value zeroBasedIndex2 = builder_.create<mlir::arith::SubIOp>(loc_, indexValAsIndex2, one);
        
        // Load element from 2D array
        elemVal = builder_.create<mlir::memref::LoadOp>(loc_, arrVarInfo->value, mlir::ValueRange{zeroBasedIndex, zeroBasedIndex2});
    } else {
        // Load element from 1D array
        elemVal = builder_.create<mlir::memref::LoadOp>(loc_, arrVarInfo->value, mlir::ValueRange{zeroBasedIndex});
    }

    // Create scalar VarInfo with element's CompleteType.
    // For homogeneous arrays, element type is the single subtype.
    if (arrVarInfo->type.subTypes.size() != 1) {
        throw std::runtime_error("ArrayAccessNode: array/matrix type must have exactly one element subtype");
    }
    CompleteType elemType = arrVarInfo->type.subTypes[0];
    VarInfo out(elemType);
    out.value = elemVal;
    out.isLValue = false;
    pushValue(out);
}
void MLIRGen::visit(ArrayTypedDecNode *node) {
    //Resolve declared variable
    VarInfo* declaredVar = currScope_->resolveVar(node->id, node->line);
    if (!declaredVar) {
        throw std::runtime_error("ArrayTypedDec node: variable not declared in scope");
    }

    // Check if array is dynamic (dims[0] == -1)
    bool isDynamic = !declaredVar->type.dims.empty() && declaredVar->type.dims[0] < 0;

    if (node->init) {
        // Vectors are handled separately - they're always allocated with zero length
        // and then resized during assignment
        if (declaredVar->type.baseType == BaseType::VECTOR) {
            // Allocate vector with zero length (will be resized during assignment)
            if (!declaredVar->value) {
                allocaVar(declaredVar, node->line);
            }
            
            // Visit initializer and assign (assignment will resize the vector)
            node->init->accept(*this);
            VarInfo literal = popValue();

            assignTo(&literal, declaredVar, node->line);
            return;
        }
        
        
        // For dynamic arrays compute size from initializer first
        if (isDynamic) {
            // Visit initializer to get source VarInfo
            node->init->accept(*this);
            VarInfo literal = popValue();
            
            // Compute size as runtime computation (in current block)
            mlir::Value sizeValue = computeArraySize(&literal, node->line);
            
            // Allocate variable with computed size (in current block where size is computed)
            // This ensures sizeValue dominates the allocation
            // For variable declarations, this should happen early enough to dominate later uses
            if (!declaredVar->value) {
                allocaVar(declaredVar, node->line, sizeValue);
            }
            
            // Assign the initializer to the variable
            assignTo(&literal, declaredVar, node->line);
            return;
        } else {
            // Static array - allocate, then assign
            // Allocate storage if not already done
            if (!declaredVar->value) {
                allocaVar(declaredVar, node->line);
            }
            
            node->init->accept(*this);
            VarInfo literal = popValue();
            assignTo(&literal, declaredVar, node->line);
            return;
        }
    }

    // Handle vectors without initializers - just allocate with zero length
    if (!node->init && declaredVar->type.baseType == BaseType::VECTOR) {
        if (!declaredVar->value) {
            allocaVar(declaredVar, node->line);
        }
        return;
    }

    // --- Zero-initialize array if no initializer and static size known ---
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

    if (allStatic) {
        if (!declaredVar->value) {
            allocaVar(declaredVar, node->line);
        }
        
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
        if (declaredVar->type.dims.size() == 2) {
            int rows = declaredVar->type.dims[0];
            int cols = declaredVar->type.dims[1];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    mlir::Value idxR = builder_.create<mlir::arith::ConstantOp>(
                        loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(r)));
                    mlir::Value idxC = builder_.create<mlir::arith::ConstantOp>(
                        loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(c)));
                    builder_.create<mlir::memref::StoreOp>(
                        loc_, zeroVal, declaredVar->value, mlir::ValueRange{idxR, idxC});
                }
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                mlir::Value idx = builder_.create<mlir::arith::ConstantOp>(
                    loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(i)));
                builder_.create<mlir::memref::StoreOp>(
                    loc_, zeroVal, declaredVar->value, mlir::ValueRange{idx});
            }
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
    VarInfo arrVarInfo(node->type);
    
    // Validate array literal type has dimensions set
    if (arrVarInfo.type.dims.empty() || arrVarInfo.type.dims[0] < 0) {
        throw std::runtime_error("ArrayLiteralNode: array literal type must have static dimensions set. dims.size()=" + 
                                 std::to_string(arrVarInfo.type.dims.size()) + 
                                 (arrVarInfo.type.dims.empty() ? "" : ", dims[0]=" + std::to_string(arrVarInfo.type.dims[0])));
    }
    
    allocaLiteral(&arrVarInfo, node->line);

    if (!arrVarInfo.value) {
        throw std::runtime_error("ArrayLiteralNode: failed to allocate array storage.");
    }

    // Empty literal: nothing to store, just return allocated container.
    if (!node->list || node->list->list.empty()) {
        pushValue(arrVarInfo);
        return;
    }

    // Handle 2D array/matrix literals
    if (node->type.dims.size() == 2) {
        CompleteType elemType = arrVarInfo.type.subTypes.empty()
            ? CompleteType(BaseType::UNKNOWN)
            : arrVarInfo.type.subTypes[0];
        
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
                VarInfo promoted = promoteType(&elem, &elemType, node->line);
                mlir::Value val = getSSAValue(promoted);
                
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
        CompleteType elemType = arrVarInfo.type.subTypes.empty()
            ? CompleteType(BaseType::UNKNOWN)
            : arrVarInfo.type.subTypes[0];
        for (size_t i = 0; i < node->list->list.size(); ++i) {
            node->list->list[i]->accept(*this);
            VarInfo elem = popValue();
            
            VarInfo promoted = promoteType(&elem, &elemType, node->line);
            mlir::Value val = getSSAValue(promoted);
            
            if (!val) {
                throw std::runtime_error("ArrayLiteralNode: element value is null at index " + std::to_string(i));
            }
            
            if (!arrVarInfo.value) {
                throw std::runtime_error("ArrayLiteralNode: array storage is null at index " + std::to_string(i));
            }
            
            // MLIR memref uses 0-based index
            auto idxConst = builder_.create<mlir::arith::ConstantOp>(
                loc_, builder_.getIndexType(), builder_.getIntegerAttr(builder_.getIndexType(), static_cast<int64_t>(i))
            );
            
            if (!idxConst) {
                throw std::runtime_error("ArrayLiteralNode: failed to create index constant at index " + std::to_string(i));
            }
            
            builder_.create<mlir::memref::StoreOp>(loc_, val, arrVarInfo.value, mlir::ValueRange{idxConst});
        }
    }
    
    pushValue(arrVarInfo);
}

void MLIRGen::visit(RangeExprNode *node)          { std::cout << "RangeExprNode not implemented\n"; }
