/*
Traverse AST tree and for each node emit MLIR operations
Backend sets up MLIR context, builder, and helper functions
After generating the MLIR, Backend will lower the dialects and output LLVM IR
*/
#include "MLIRgen.h"
#include "BackEnd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


#include <stdexcept>

MLIRGen::MLIRGen(BackEnd& backend)
    : backend_(backend),
      builder_(*backend.getBuilder()),
      module_(backend.getModule()),
      context_(backend.getContext()),
      loc_(backend.getLoc()) {}

MLIRGen::MLIRGen(BackEnd& backend, Scope* rootScope)
    : backend_(backend),
      builder_(*backend.getBuilder()),
      module_(backend.getModule()),
      context_(backend.getContext()),
      loc_(backend.getLoc()),
      root_(rootScope) {}

mlir::Value MLIRGen::popValue() {
    if (v_stack_.empty()) {
        throw std::runtime_error("MLIRGen internal error: value stack underflow.");
    }
    mlir::Value value = v_stack_.back();
    v_stack_.pop_back();
    return value;
}

void MLIRGen::pushValue(mlir::Value value) {
    if (!value) {
        throw std::runtime_error("MLIRGen internal error: attempting to push empty value onto stack.");
    }
    v_stack_.push_back(value);
}

void MLIRGen::visit(FileNode* node) {
    // Initialize current scope to the semantic root if provided
    currScope_ = root_;
    for (auto& line: node->stats) {
        line->accept(*this);
    }
}

void MLIRGen::visit(TrueNode* node) {
    auto boolType = builder_.getI1Type();

    auto trueLiteral = builder_.create<mlir::arith::ConstantOp>(
        loc_, boolType, builder_.getIntegerAttr(boolType, 1)
    );
    pushValue(trueLiteral);
}


void MLIRGen::visit(FalseNode* node) {
    auto boolType = builder_.getI1Type();

    auto trueLiteral = builder_.create<mlir::arith::ConstantOp>(
        loc_, boolType, builder_.getIntegerAttr(boolType, 0)
    );
    pushValue(trueLiteral);
}

void MLIRGen::visit(CharNode* node) {
    auto charType = builder_.getI8Type();

    auto charLiteral = builder_.create<mlir::arith::ConstantOp>(
        loc_, charType, builder_.getIntegerAttr(charType, static_cast<int>(node->value))
    );
    pushValue(charLiteral);
}

void MLIRGen::visit(IntNode* node) {
    auto intType = builder_.getI32Type();

    auto intLiteral = builder_.create<mlir::arith::ConstantOp>(
        loc_, intType, builder_.getIntegerAttr(intType, node->value)
    );
    pushValue(intLiteral);
}

void MLIRGen::visit(RealNode* node) {
    auto realType = builder_.getF32Type();

    auto realLiteral = builder_.create<mlir::arith::ConstantOp>(
        loc_, realType, builder_.getFloatAttr(realType, node->value)
    );
    pushValue(realLiteral);
}

void MLIRGen::visit(IdNode* node) {
    
}


void MLIRGen::visit(ParentExpr* node) {
    node->expr->accept(*this);
}

void MLIRGen::visit(UnaryExpr* node) {
    node->expr->accept(*this);
    mlir::Value operand = popValue();

    switch (node->op) {
        case "+":
            pushValue(operand);
            break;
        case "-":
            //subtract from zero works for both int and real
            auto zero = builder_.create<mlir::arith::ConstantOp>(
                loc_, operand.getType(), builder_.getZeroAttr(operand.getType()));
            auto neg = builder_.create<mlir::arith::SubIOp>(loc_, zero, operand);
            pushValue(neg);
            break;
    }
}

void MLIRGen::visit(MultExpr* node){
    node->left->accept(*this);
    mlir::Value left = popValue();
    node->right->accept(*this);
    mlir::Value right = popValue();

    if(left.getType().isa<mlir::IntegerType>()) {
        if (node->op == "*") {
            auto mul = builder_.create<mlir::arith::MulIOp>(loc_, left, right);
            pushValue(mul);
        } else if (node->op == "/") {
            auto div = builder_.create<mlir::arith::DivSIOp>(loc_, left, right);
            pushValue(div);
        } else if (node->op == "%") {
            auto rem = builder_.create<mlir::arith::RemSIOp>(loc_, left, right);
            pushValue(rem);
        }
    } else if(left.getType().isa<mlir::FloatType>()) {
        if (node->op == "*") {
            auto mul = builder_.create<mlir::arith::MulFOp>(loc_, left, right);
            pushValue(mul);
        } else if (node->op == "/") {
            auto div = builder_.create<mlir::arith::DivFOp>(loc_, left, right);
            pushValue(div);
        }
    } else {
        throw std::runtime_error("MLIRGen Error: Unsupported type for multiplication.");
    }
}

void MLIRGen::visit(AddExpr* node){
    node->left->accept(*this);
    mlir::Value left = popValue();
    node->right->accept(*this);
    mlir::Value right = popValue();

    if(left.getType().isa<mlir::IntegerType>()) {
        if (node->op == "+") {
            auto add = builder_.create<mlir::arith::AddIOp>(loc_, left, right);
            pushValue(add);
        } else if (node->op == "-") {
            auto sub = builder_.create<mlir::arith::SubIOp>(loc_, left, right);
            pushValue(sub);
        }
    } else if(left.getType().isa<mlir::FloatType>()) {
        if (node->op == "+") {
            auto add = builder_.create<mlir::arith::AddFOp>(loc_, left, right);
            pushValue(add);
        } else if (node->op == "-") {
            auto sub = builder_.create<mlir::arith::SubFOp>(loc_, left, right);
            pushValue(sub);
        }
    } else {
        throw std::runtime_error("MLIRGen Error: Unsupported type for addition.");
    }
}

void MLIRGen::visit(NotExpr* node) {
    node->expr->accept(*this);
    mlir::Value operand = popValue();

    auto one = builder_.create<mlir::arith::ConstantOp>(
        loc_, operand.getType(), builder_.getIntegerAttr(operand.getType(), 1));
    auto notOp = builder_.create<mlir::arith::XOrIOp>(loc_, operand, one);
    pushValue(notOp);
}

void MLIRGen::visit(EqExpr* node){
    node->left->accept(*this);
    mlir::Value left = popValue();
    node->right->accept(*this);
    mlir::Value right = popValue();

    mlir::Type type = left.getType();
    mlir::Value = eqResult;
   
    if (type.isa<mlir::TupleType>()) {
        eqResult = mlirTupleEquals(left, right, loc_, builder_);
    } else {
        eqResult = mlirScalarEquals(left, right, loc_, builder_);
    }

    if (node->op == "==") {
        pushValue(eqResult);
    } else if (node->op == "!=") {
        auto one = builder_.create<mlir::arith::ConstantOp>(loc_, eqResult.getType(), builder_.getIntegerAttr(eqResult.getType(), 1));
        auto notEq = builder_.create<mlir::arith::XOrIOp>(loc_, eqResult, one);
        pushValue(notEq);
    }
}


// Helper functions for equality
mlir::Value scalarEquals(mlir::Value left, mlir::Value right, mlir::Location loc, mlir::OpBuilder& builder) {
    mlir::Type type = left.getType();
    if (type.isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, left, right);
    } else if (type.isa<mlir::FloatType>()) {
        return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, left, right);
    } else {
        throw std::runtime_error("mlirScalarEquals: Unsupported type for equality");
    }
}

mlir::Value mlirTupleEquals(mlir::Value left, mlir::Value right, mlir::Location loc, mlir::OpBuilder& builder) {
    // left and right are both of TupleType
    // otherwise, error
    auto tupleType = left.getType().cast<mlir::TupleType>();
    int numElements = tupleType.size();
    mlir::Value result;

    for (int i = 0; i < numElements; ++i) {
        auto leftElem = builder.create<mlir::TupleGetOp>(loc, left, i);
        auto rightElem = builder.create<mlir::TupleGetOp>(loc, right, i);
        mlir::Type elemType = tupleType.getType(i);

        mlir::Value elemEq;
        if (elemType.isa<mlir::TupleType>()) {
            elemEq = mlirTupleEquals(leftElem, rightElem, loc, builder); // recursive for nested tuples
        } else {
            elemEq = mlirScalarEquals(leftElem, rightElem, loc, builder);
        }

        if (i == 0) {
            result = elemEq;
        } else {
            result = builder.create<mlir::arith::AndIOp>(loc, result, elemEq);
        }
    }
    return result;
}


void MLIRGen::visit(AndExpr* node){
    node->left->accept(*this);
    mlir::Value left = popValue();
    node->right->accept(*this);
    mlir::Value right = popValue();

    auto andOp = builder_.create<mlir::arith::AndIOp>(loc_, left, right);
    pushValue(andOp);
}

void MLIRGen::visit(OrExpr* node){
    node->left->accept(*this);
    mlir::Value left = popValue();
    node->right->accept(*this);
    mlir::Value right = popValue();

    if(node->op == "or") {
        auto orOp = builder_.create<mlir::arith::OrIOp>(loc_, left, right);
        pushValue(orOp);
    } else if (node->op == "xor") {
        auto xorOp = builder_.create<mlir::arith::XOrIOp>(loc_, left, right);
        pushValue(xorOp);
    }
}