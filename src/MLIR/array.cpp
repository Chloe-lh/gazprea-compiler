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
    
    // 1. Resolve source array variable
    VarInfo* arrVarInfo = currScope_->resolveVar(node->id, node->line);
    if (!arrVarInfo) {
        throw std::runtime_error("ArraySliceExpr: unresolved array '" + node->id + "'");
    }
    if (arrVarInfo->type.baseType != BaseType::ARRAY) {
        throw std::runtime_error("ArraySliceExpr: Variable '" + node->id + "' is not an array.");
    }
    
    // Ensure array is allocated
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
    
    // 2. Extract pointer from memref
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
    mlir::Value basePtr;
    
    if (arrVarInfo->value.getType().isa<mlir::MemRefType>()) {
        // Extract the base pointer from the memref descriptor
        mlir::Value baseIndex = builder_.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
            loc_, arrVarInfo->value
        );
        
        // Convert index to i64, then to LLVM pointer
        auto i64Ty = builder_.getI64Type();
        mlir::Value ptrInt = builder_.create<mlir::arith::IndexCastOp>(
            loc_, i64Ty, baseIndex
        );
        
        basePtr = builder_.create<mlir::LLVM::IntToPtrOp>(loc_, ptrTy, ptrInt);
    } else {
        throw std::runtime_error("ArraySliceExpr: array value is not a memref");
    }
    
    // 3. Get array length
    auto indexTy = builder_.getIndexType();
    auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    mlir::Value baseLen = builder_.create<mlir::memref::DimOp>(loc_, arrVarInfo->value, zeroIdx);
    
    // 4. Evaluate and convert start expression
    mlir::Value start_0based;
    if (node->range && node->range->start) {
        // Visit start expression
        node->range->start->accept(*this);
        VarInfo startVarInfo = popValue();
        mlir::Value startVal = getSSAValue(startVarInfo);
        
        // Convert to index type (signed)
        mlir::Value startIndex = builder_.create<mlir::arith::IndexCastOp>(
            loc_, indexTy, startVal
        );
        
        // Check if negative and convert if needed
        auto zeroIndex = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        auto isNegative = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::slt, startIndex, zeroIndex
        );
        
        auto oneIndex = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
        
        // If negative: start_1based = baseLen + start + 1
        // If non-negative: start_1based = start
        auto ifOpStart = builder_.create<mlir::scf::IfOp>(loc_, indexTy, isNegative, true);
        
        // Then: use negative conversion
        builder_.setInsertionPointToStart(&ifOpStart.getThenRegion().front());
        mlir::Value start_1based_neg = builder_.create<mlir::arith::AddIOp>(
            loc_, baseLen, startIndex
        );
        start_1based_neg = builder_.create<mlir::arith::AddIOp>(
            loc_, start_1based_neg, oneIndex
        );
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{start_1based_neg});
        
        // Else: use as-is
        builder_.setInsertionPointToStart(&ifOpStart.getElseRegion().front());
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{startIndex});
        
        builder_.setInsertionPointAfter(ifOpStart);
        mlir::Value start_1based = ifOpStart.getResult(0);
        
        // Convert from 1-based to 0-based: start_0based = start_1based - 1
        start_0based = builder_.create<mlir::arith::SubIOp>(
            loc_, start_1based, oneIndex
        );
    } else {
        // No start specified (..end case): start = 0 (0-based)
        start_0based = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    }
    
    // 5. Evaluate and convert end expression
    mlir::Value end_0based;
    if (node->range && node->range->end) {
        // Visit end expression
        node->range->end->accept(*this);
        VarInfo endVarInfo = popValue();
        mlir::Value endVal = getSSAValue(endVarInfo);
        
        // Convert to index type (signed)
        mlir::Value endIndex = builder_.create<mlir::arith::IndexCastOp>(
            loc_, indexTy, endVal
        );
        
        // Check if negative and convert if needed
        auto zeroIndex = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        auto isNegative = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::slt, endIndex, zeroIndex
        );
        
        // If negative: end_1based = baseLen + end + 1
        // If non-negative: end_1based = end
        auto ifOpEnd = builder_.create<mlir::scf::IfOp>(loc_, indexTy, isNegative, true);
        
        // Then: use negative conversion
        builder_.setInsertionPointToStart(&ifOpEnd.getThenRegion().front());
        auto oneIndex = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
        mlir::Value end_1based_neg = builder_.create<mlir::arith::AddIOp>(
            loc_, baseLen, endIndex
        );
        end_1based_neg = builder_.create<mlir::arith::AddIOp>(
            loc_, end_1based_neg, oneIndex
        );
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{end_1based_neg});
        
        // Else: use as-is
        builder_.setInsertionPointToStart(&ifOpEnd.getElseRegion().front());
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{endIndex});
        
        builder_.setInsertionPointAfter(ifOpEnd);
        mlir::Value end_1based = ifOpEnd.getResult(0);
        
        // Convert from 1-based to 0-based: end_0based = end_1based - 1
        // Note: end is exclusive in Gazprea, so we subtract 1
        auto oneIndex2 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
        end_0based = builder_.create<mlir::arith::SubIOp>(
            loc_, end_1based, oneIndex2
        );
    } else {
        // No end specified (start.. case): end = baseLen (already 0-based)
        end_0based = baseLen;
    }
    
    // 6. Convert indices to i64 for runtime call
    auto i64Ty = builder_.getI64Type();
    mlir::Value baseLen_i64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, baseLen);
    mlir::Value start_i64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, start_0based);
    mlir::Value end_i64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, end_0based);
    
    // 7. Call runtime function
    auto sliceFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("gaz_slice_int_checked");
    if (!sliceFunc) {
        throw std::runtime_error("ArraySliceExpr: gaz_slice_int_checked function not found");
    }
    
    auto callOp = builder_.create<mlir::LLVM::CallOp>(
        loc_, sliceFunc, mlir::ValueRange{basePtr, baseLen_i64, start_i64, end_i64}
    );
    mlir::Value sliceStruct = callOp.getResult();
    
    // 8. Create VarInfo for slice result
    CompleteType sliceType = arrVarInfo->type;
    sliceType.dims.clear();
    sliceType.dims.push_back(-1); // Dynamic dimension
    
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

    // Visit index expression
    target->indexExpr->accept(*this);
    VarInfo indexVarInfo = popValue();
    mlir::Value indexVal = getSSAValue(indexVarInfo);

    // Convert to index type and adjust for 0-based indexing
    auto idxTy = builder_.getIndexType();
    mlir::Value indexValAsIndex = builder_.create<mlir::arith::IndexCastOp>(loc_, idxTy, indexVal);
    auto one = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    mlir::Value zeroBasedIndex = builder_.create<mlir::arith::SubIOp>(loc_, indexValAsIndex, one);

    // Build index and store
    builder_.create<mlir::memref::StoreOp>(
        loc_, val, arrayVar->value, mlir::ValueRange{zeroBasedIndex}
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

    // Visit index expression
    node->indexExpr->accept(*this);
    VarInfo indexVarInfo = popValue();
    mlir::Value indexVal = getSSAValue(indexVarInfo);

    // Convert to index type and adjust for 0-based indexing
    auto idxTy = builder_.getIndexType();
    mlir::Value indexValAsIndex = builder_.create<mlir::arith::IndexCastOp>(loc_, idxTy, indexVal);
    auto one = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    mlir::Value zeroBasedIndex = builder_.create<mlir::arith::SubIOp>(loc_, indexValAsIndex, one);

    // memref.load element
    mlir::Value elemVal = builder_.create<mlir::memref::LoadOp>(loc_, arrVarInfo->value, mlir::ValueRange{zeroBasedIndex});

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

    // Check if array is dynamic (dims[0] == -1)
    bool isDynamic = !declaredVar->type.dims.empty() && declaredVar->type.dims[0] < 0;

    if (node->init) {
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
            
            node->init->accept(*this);
            VarInfo literal = popValue();
            assignTo(&literal, declaredVar, node->line);
            return;
        }
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

    VarInfo arrVarInfo(node->type);
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
