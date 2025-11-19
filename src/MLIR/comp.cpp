#include "MLIRgen.h"
#include "MLIR/helpers/compHelper.h"


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
    VarInfo leftInfo = popValue();
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    
    auto loadIfMemref = [&](mlir::Value v) -> mlir::Value {
        if (v.getType().isa<mlir::MemRefType>()) {
            return builder_.create<mlir::memref::LoadOp>(loc_, v, mlir::ValueRange{});
        }
        return v;
    };

    mlir::Value left = loadIfMemref(leftInfo.value);
    mlir::Value right = loadIfMemref(rightInfo.value);

    mlir::Value result;
    if(node->op == "or") {
        result = builder_.create<mlir::arith::OrIOp>(loc_, left, right);
    } else if (node->op == "xor") {
        result = builder_.create<mlir::arith::XOrIOp>(loc_, left, right);
    }

    // Wrap result into memref-backed VarInfo
    VarInfo outVar(leftInfo.type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}

void MLIRGen::visit(AndExpr* node){
    if (tryEmitConstantForNode(node)) return;
    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    node->right->accept(*this);
    VarInfo rightInfo = popValue();

    auto loadIfMemref = [&](mlir::Value v) -> mlir::Value {
        if (v.getType().isa<mlir::MemRefType>()) {
            return builder_.create<mlir::memref::LoadOp>(loc_, v, mlir::ValueRange{});
        }
        return v;
    };
    
    mlir::Value left = loadIfMemref(leftInfo.value);
    mlir::Value right = loadIfMemref(rightInfo.value);

    auto andOp = builder_.create<mlir::arith::AndIOp>(loc_, left, right);
    
    // Wrap boolean result into memref-backed VarInfo
    VarInfo outVar(leftInfo.type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, andOp, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}


void MLIRGen::visit(EqExpr* node){
    if (tryEmitConstantForNode(node)) return;

    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    node->right->accept(*this);
    VarInfo rightInfo = popValue();

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
        result = builder_.create<mlir::arith::XOrIOp>(loc_, result, one);
    }

    VarInfo outVar(node->type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}


void MLIRGen::visit(NotExpr* node) {
    if (tryEmitConstantForNode(node)) return;
    node->operand->accept(*this);
    VarInfo operandInfo = popValue();
    // Load scalar if value is a memref
    mlir::Value operand = operandInfo.value;
    if (operand.getType().isa<mlir::MemRefType>()) {
        operand = builder_.create<mlir::memref::LoadOp>(loc_, operandInfo.value, mlir::ValueRange{});
    }

    auto one = builder_.create<mlir::arith::ConstantOp>(
        loc_, operand.getType(), builder_.getIntegerAttr(operand.getType(), 1));
    auto notOp = builder_.create<mlir::arith::XOrIOp>(loc_, operand, one);

    // Store result into memref-backed VarInfo and push
    VarInfo outVar(operandInfo.type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, notOp, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}