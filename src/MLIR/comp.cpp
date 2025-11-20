#include "MLIRgen.h"
#include "MLIR/helpers/compHelper.h"
#include "Scope.h"
#include "Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <iostream>


void MLIRGen::visit(CompExpr* node) {
    if (tryEmitConstantForNode(node)) return;
    
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
    VarInfo leftPromoted = castType(&leftVarInfo, &promotedType);
    VarInfo rightPromoted = castType(&rightVarInfo, &promotedType);
    
    // Load the values
    mlir::Value leftVal = builder_.create<mlir::memref::LoadOp>(
        loc_, leftPromoted.value, mlir::ValueRange{}
    );
    mlir::Value rightVal = builder_.create<mlir::memref::LoadOp>(
        loc_, rightPromoted.value, mlir::ValueRange{}
    );
    
    // Create comparison operation based on operator and type
    mlir::Value cmpResult;
    CompleteType boolType = CompleteType(BaseType::BOOL);
    VarInfo resultVarInfo = VarInfo(boolType);
    allocaLiteral(&resultVarInfo);
    
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
        );
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
        );
    } else {
        throw std::runtime_error("CompExpr: comparison not supported for type");
    }
    
    // Store the comparison result
    builder_.create<mlir::memref::StoreOp>(loc_, cmpResult, resultVarInfo.value, mlir::ValueRange{});
    
    // Push result onto stack
    pushValue(resultVarInfo);
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
    
    VarInfo leftPromoted = castType(&leftInfo, &promotedType);
    VarInfo rightPromoted = castType(&rightInfo, &promotedType);

    mlir::Value result;

    if (promotedType.baseType == BaseType::TUPLE) {
        // Load tuple structs from their pointers and compare element-wise.
        mlir::Type structTy = getLLVMType(promotedType);
        mlir::Value leftStruct = builder_.create<mlir::LLVM::LoadOp>(
            loc_, structTy, leftPromoted.value);
        mlir::Value rightStruct = builder_.create<mlir::LLVM::LoadOp>(
            loc_, structTy, rightPromoted.value);
        result = mlirAggregateEquals(leftStruct, rightStruct, loc_, builder_);
    } else {
        // Load the scalar values from their memrefs
        mlir::Value left = builder_.create<mlir::memref::LoadOp>(
            loc_, leftPromoted.value, mlir::ValueRange{}
        );
        mlir::Value right = builder_.create<mlir::memref::LoadOp>(
            loc_, rightPromoted.value, mlir::ValueRange{}
        );
        result = mlirScalarEquals(left, right, loc_, builder_);
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