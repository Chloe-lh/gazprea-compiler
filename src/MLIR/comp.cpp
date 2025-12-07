#include "MLIRgen.h"
#include "MLIR/helpers/compHelper.h"
#include "Scope.h"
#include "Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <iostream>


void MLIRGen::visit(CompExpr* node) {
    // if (tryEmitConstantForNode(node)) return;
    
    // Visit left and right operands
    node->left->accept(*this);
    node->right->accept(*this);
    
    // Pop operands (right first, then left)
    VarInfo rightVarInfo = popValue();
    VarInfo leftVarInfo = popValue();
    
    // Determine the promoted type for comparison
    CompleteType promotedType = promote(leftVarInfo.type, rightVarInfo.type);
    if (promotedType.baseType == BaseType::UNKNOWN) {
        promotedType = promote(rightVarInfo.type, leftVarInfo.type);
    }
    if (promotedType.baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("CompExpr: cannot promote types for comparison");
    }
    
    // Cast both operands to the promoted type
    VarInfo leftPromoted = castType(&leftVarInfo, &promotedType, node->line);
    VarInfo rightPromoted = castType(&rightVarInfo, &promotedType, node->line);
    
    // Load the values. Prefer an explicit memref.load for variables so we
    // always read the current stored value; fall back to SSA value when
    // the VarInfo already holds an SSA value.
    mlir::Value leftVal;
    if (leftPromoted.value && leftPromoted.value.getType().isa<mlir::MemRefType>()) {
        leftVal = builder_.create<mlir::memref::LoadOp>(loc_, leftPromoted.value, mlir::ValueRange{}).getResult();
    } else {
        leftVal = leftPromoted.value;
    }

    mlir::Value rightVal;
    if (rightPromoted.value && rightPromoted.value.getType().isa<mlir::MemRefType>()) {
        rightVal = builder_.create<mlir::memref::LoadOp>(loc_, rightPromoted.value, mlir::ValueRange{}).getResult();
    } else {
        rightVal = rightPromoted.value;
    }
    
    // Create comparison operation based on operator and type
    mlir::Value cmpResult;
    
    if (promotedType.baseType == BaseType::INTEGER) {
        // Integer comparison
        mlir::arith::CmpIPredicate predicate;
        if (node->op == "<") {
            predicate = mlir::arith::CmpIPredicate::slt;
        } else if (node->op == ">") {
            predicate = mlir::arith::CmpIPredicate::sgt;
        } else if (node->op == "<=") {
            predicate = mlir::arith::CmpIPredicate::sle;
        } else if (node->op == ">=") {
            predicate = mlir::arith::CmpIPredicate::sge;
        } else if (node->op == "==") {
            predicate = mlir::arith::CmpIPredicate::eq;
        } else if (node->op == "!=") {
            predicate = mlir::arith::CmpIPredicate::ne;
        } else {
            throw std::runtime_error("CompExpr: unknown operator '" + node->op + "'");
        }
        cmpResult = builder_.create<mlir::arith::CmpIOp>(
            loc_, predicate, leftVal, rightVal
        ).getResult();
    } else if (promotedType.baseType == BaseType::REAL) {
        // Floating point comparison
        mlir::arith::CmpFPredicate predicate;
        if (node->op == "<") {
            predicate = mlir::arith::CmpFPredicate::OLT;
        } else if (node->op == ">") {
            predicate = mlir::arith::CmpFPredicate::OGT;
        } else if (node->op == "<=") {
            predicate = mlir::arith::CmpFPredicate::OLE;
        } else if (node->op == ">=") {
            predicate = mlir::arith::CmpFPredicate::OGE;
        } else if (node->op == "==") {
            predicate = mlir::arith::CmpFPredicate::OEQ;
        } else if (node->op == "!=") {
            predicate = mlir::arith::CmpFPredicate::ONE;
        } else {
            throw std::runtime_error("CompExpr: unknown operator '" + node->op + "'");
        }
        cmpResult = builder_.create<mlir::arith::CmpFOp>(
            loc_, predicate, leftVal, rightVal
        ).getResult();
    } else {
        throw std::runtime_error("CompExpr: comparison not supported for type");
    }
    VarInfo outVar{CompleteType(BaseType::BOOL)};
    outVar.value = cmpResult;
    outVar.isLValue = false;
    pushValue(outVar);
}

void MLIRGen::visit(OrExpr* node){
    if (tryEmitConstantForNode(node)) return;
    node->left->accept(*this);
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    VarInfo leftInfo = popValue();

    mlir::Value lhs = getSSAValue(leftInfo);
    mlir::Value rhs = getSSAValue(rightInfo);


    VarInfo result = (CompleteType(BaseType::BOOL));
    mlir::Value compSSA;
    result.isLValue = false;
    if(node->op=="or"){
        compSSA = builder_.create<mlir::arith::OrIOp>(loc_, lhs, rhs);
    }else if(node->op=="xor"){
        compSSA = builder_.create<mlir::arith::XOrIOp>(loc_, lhs, rhs);
    }
    result.value = compSSA;
    pushValue(result);
}

void MLIRGen::visit(AndExpr* node){
    if (tryEmitConstantForNode(node)) { return; }
    node->left->accept(*this);
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    VarInfo leftInfo = popValue();

    // no need to use memref
    mlir::Value lhs = getSSAValue(leftInfo);
    mlir::Value rhs = getSSAValue(rightInfo);

    auto andSSA = builder_.create<mlir::arith::AndIOp>(loc_, lhs, rhs).getResult();

    VarInfo result = (CompleteType(BaseType::BOOL));
    result.value = andSSA;
    result.isLValue = false;
    pushValue(result);
}


void MLIRGen::visit(EqExpr* node){
    if (tryEmitConstantForNode(node)) return;

    node->left->accept(*this);
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    VarInfo leftInfo = popValue();

    CompleteType promotedType = promote(leftInfo.type, rightInfo.type);
    if (promotedType.baseType == BaseType::UNKNOWN) {
        promotedType = promote(rightInfo.type, leftInfo.type);
    }
    if (promotedType.baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("EqExpr: cannot promote types for comparison");
    }
    
    VarInfo leftPromoted = castType(&leftInfo, &promotedType, node->line);
    VarInfo rightPromoted = castType(&rightInfo, &promotedType, node->line);

    mlir::Value result;

    if (promotedType.baseType == BaseType::TUPLE ||
        promotedType.baseType == BaseType::STRUCT) {
        // Load aggregate structs from their pointers and compare element-wise.
        mlir::Type structTy = getLLVMType(promotedType);
        mlir::Value leftStruct = builder_.create<mlir::LLVM::LoadOp>(
            loc_, structTy, leftPromoted.value);
        mlir::Value rightStruct = builder_.create<mlir::LLVM::LoadOp>(
            loc_, structTy, rightPromoted.value);
        result = mlirAggregateEquals(leftStruct, rightStruct, loc_, builder_);
    } else if (
        promotedType.baseType == BaseType::ARRAY || 
        promotedType.baseType == BaseType::VECTOR ||
        promotedType.baseType == BaseType::MATRIX
    ) {
        // Ensure storage exists
        if (!leftPromoted.value) allocaVar(&leftPromoted, node->line);
        if (!rightPromoted.value) allocaVar(&rightPromoted, node->line);

        // Ensure runtime dimensions are populated
        if (leftPromoted.runtimeDims.empty()) leftPromoted.runtimeDims = promotedType.dims;
        if (rightPromoted.runtimeDims.empty()) rightPromoted.runtimeDims = promotedType.dims;

        auto &lhsDims = leftPromoted.runtimeDims;
        auto &rhsDims = rightPromoted.runtimeDims;

        auto i1Ty = builder_.getI1Type();

        // Dimension checks
        if (lhsDims.size() != rhsDims.size() || lhsDims.empty() || rhsDims.empty()) throw std::runtime_error("MLIRGen::EqExpr: Comparison with two elements of dim size " + std::to_string(lhsDims.size()) + " and " + std::to_string(rhsDims.size()));
        if (lhsDims[0] != rhsDims[0]) throw std::runtime_error("MLIRGen::EqExpr: Comparison with two elements of dim size " + std::to_string(lhsDims[0]) + " and " + std::to_string(rhsDims[0]));

        // Handle vector / arrays
        if (lhsDims.size() == 1) {
            int64_t lhsLen = lhsDims[0];
            int64_t rhsLen = rhsDims[0];

            // Accumulator stored in a temporary boolean memref
            VarInfo accInfo{CompleteType(BaseType::BOOL)};
            allocaLiteral(&accInfo, node->line);

            // Assume true at the start
            mlir::Value trueVal = builder_.create<mlir::arith::ConstantOp>(loc_, i1Ty, builder_.getIntegerAttr(i1Ty, 1)).getResult();
            builder_.create<mlir::memref::StoreOp>(loc_, trueVal, accInfo.value, mlir::ValueRange{});

            auto idxTy = builder_.getIndexType();
            mlir::Value lb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 0));
            mlir::Value ub = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, lhsLen));
            mlir::Value step = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 1));

            auto forOp = builder_.create<mlir::scf::ForOp>(loc_, lb, ub, step, mlir::ValueRange{});

            builder_.setInsertionPointToStart(forOp.getBody());
            {
                mlir::Value idx = forOp.getInductionVar();

                mlir::Value lElem = builder_.create<mlir::memref::LoadOp>(loc_, leftPromoted.value, mlir::ValueRange{idx});
                mlir::Value rElem = builder_.create<mlir::memref::LoadOp>(loc_, rightPromoted.value, mlir::ValueRange{idx});

                mlir::Type elemTy = lElem.getType();
                mlir::Value elemEq;
                if (elemTy.isa<mlir::LLVM::LLVMStructType>()) {elemEq = mlirAggregateEquals(lElem, rElem, loc_, builder_);
                } else {
                    elemEq = mlirScalarEquals(lElem, rElem, loc_, builder_);
                }

                mlir::Value accVal = builder_.create<mlir::memref::LoadOp>(loc_, accInfo.value, mlir::ValueRange{});
                mlir::Value newAcc = builder_.create<mlir::arith::AndIOp>(loc_, accVal, elemEq);
                builder_.create<mlir::memref::StoreOp>(loc_, newAcc, accInfo.value, mlir::ValueRange{});
            }

            builder_.setInsertionPointAfter(forOp);
            result = builder_.create<mlir::memref::LoadOp>(loc_, accInfo.value, mlir::ValueRange{}).getResult();

        } else if (lhsDims.size() == 2) {
            // 2D equality (matrix / 2D array)
            int64_t lhsRows = lhsDims[0];
            int64_t lhsCols = lhsDims[1];
            int64_t rhsRows = rhsDims[0];
            int64_t rhsCols = rhsDims[1];

            if (lhsRows != rhsRows || lhsCols != rhsCols ||
                lhsRows == 0 || lhsCols == 0) {
                throw std::runtime_error("MLIRGen::EqExpr: Invalid row/col size of " + std::to_string(lhsRows) + "x" + std::to_string(lhsCols) + " and " + std::to_string(rhsRows) + "x" + std::to_string(rhsCols));
            }

            VarInfo accInfo{CompleteType(BaseType::BOOL)};
            allocaLiteral(&accInfo, node->line);

            mlir::Value trueVal = builder_.create<mlir::arith::ConstantOp>(loc_, i1Ty, builder_.getIntegerAttr(i1Ty, 1)).getResult();
            builder_.create<mlir::memref::StoreOp>(loc_, trueVal, accInfo.value, mlir::ValueRange{});

            auto idxTy = builder_.getIndexType();

            mlir::Value rowLb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 0));
            mlir::Value rowUb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, lhsRows));
            mlir::Value one = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 1));

            auto outerFor = builder_.create<mlir::scf::ForOp>(loc_, rowLb, rowUb, one, mlir::ValueRange{});

            builder_.setInsertionPointToStart(outerFor.getBody());
            {
                mlir::Value rowIdx = outerFor.getInductionVar();

                mlir::Value colLb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 0));
                mlir::Value colUb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, lhsCols));

                auto innerFor = builder_.create<mlir::scf::ForOp>(loc_, colLb, colUb, one, mlir::ValueRange{});

                builder_.setInsertionPointToStart(innerFor.getBody());
                {
                    mlir::Value colIdx = innerFor.getInductionVar();

                    mlir::Value lElem = builder_.create<mlir::memref::LoadOp>(loc_, leftPromoted.value, mlir::ValueRange{rowIdx, colIdx});
                    mlir::Value rElem = builder_.create<mlir::memref::LoadOp>(loc_, rightPromoted.value, mlir::ValueRange{rowIdx, colIdx});

                    mlir::Type elemTy = lElem.getType();
                    mlir::Value elemEq;
                    if (elemTy.isa<mlir::LLVM::LLVMStructType>()) {
                        elemEq = mlirAggregateEquals(lElem, rElem, loc_, builder_);
                    } else {
                        elemEq = mlirScalarEquals(lElem, rElem, loc_, builder_);
                    }

                    mlir::Value accVal = builder_.create<mlir::memref::LoadOp>(loc_, accInfo.value, mlir::ValueRange{});
                    mlir::Value newAcc = builder_.create<mlir::arith::AndIOp>(loc_, accVal, elemEq);
                    builder_.create<mlir::memref::StoreOp>(loc_, newAcc, accInfo.value, mlir::ValueRange{});
                }

                builder_.setInsertionPointAfter(innerFor);

                builder_.setInsertionPointAfter(outerFor);
                result = builder_.create<mlir::memref::LoadOp>(loc_, accInfo.value, mlir::ValueRange{}).getResult();
            }
        } else {
            throw std::runtime_error("MLIRGen::EqExpr: Invalid dimensions for comparison of " + toString(promotedType));
        }
    } else if (isScalarType(promotedType.baseType)) {
        // Load the scalar values from their memrefs
        mlir::Value left = getSSAValue(leftPromoted);
        mlir::Value right = getSSAValue(rightPromoted);
        result = mlirScalarEquals(left, right, loc_, builder_);
    } else {
        throw std::runtime_error("MLIRGen::EqExpr: Unknown type '" + toString(promotedType.baseType) + "'.");
    }

    if (node->op == "!=") {
        auto one = builder_.create<mlir::arith::ConstantOp>(loc_, result.getType(), builder_.getIntegerAttr(result.getType(), 1));
        result = builder_.create<mlir::arith::XOrIOp>(loc_, result, one.getResult()).getResult();
    }

    // Prefer to keep expression results as SSA values when possible.
    // `mlirScalarEquals` / `mlirAggregateEquals` return an SSA i1 value,
    // so produce an SSA-backed VarInfo (isLValue=false) and push it.
    VarInfo outVar{CompleteType(BaseType::BOOL)};
    outVar.value = result;
    outVar.isLValue = false;
    outVar.identifier = "";
    pushValue(outVar);
}


void MLIRGen::visit(NotExpr* node) {
    if (tryEmitConstantForNode(node)) { return; }
    node->operand->accept(*this);
    VarInfo operandInfo = popValue();

    mlir::Value operandSSA = getSSAValue(operandInfo);

    // Create a constant '1' of the same integer type as the operand, then XOR
    auto oneConst = builder_.create<mlir::arith::ConstantOp>(
        loc_, operandSSA.getType(), builder_.getIntegerAttr(operandSSA.getType(), 1));
    mlir::Value notSSA = builder_.create<mlir::arith::XOrIOp>(
        loc_, operandSSA, oneConst.getResult()).getResult();
    VarInfo result = (CompleteType(BaseType::BOOL));
    result.value = notSSA;
    result.isLValue = false;
    pushValue(result);
}
