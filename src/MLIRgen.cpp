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
#include "mlir/Dialect/Math/IR/Math.h"
#include "CompileTimeExceptions.h"


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

VarInfo MLIRGen::popValue() {
    if (v_stack_.empty()) {
        throw std::runtime_error("MLIRGen internal error: value stack underflow.");
    }
    VarInfo v = std::move(v_stack_.back());
    v_stack_.pop_back();
    return v;
}

void MLIRGen::pushValue(VarInfo& value) {
    v_stack_.push_back(value);
}

void MLIRGen::visit(FileNode* node) {
    // Initialize current scope to the semantic root if provided
    currScope_ = root_;
    for (auto& line: node->stats) {
        line->accept(*this);
    }
}



// Functions
void MLIRGen::visit(FuncStatNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(FuncPrototypeNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(FuncBlockNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(ProcedureNode* node) { throw std::runtime_error("not implemented"); }

// Declarations
void MLIRGen::visit(TypedDecNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(InferredDecNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(TupleTypedDecNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(TypeAliasDecNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(TypeAliasNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(TupleTypeAliasNode* node) { throw std::runtime_error("not implemented"); }

// Statements
void MLIRGen::visit(AssignStatNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(OutputStatNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(InputStatNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(BreakStatNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(ContinueStatNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(ReturnStatNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(CallStatNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(IfNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(LoopNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(BlockNode* node) { throw std::runtime_error("not implemented"); }

// Expressions / Operators
void MLIRGen::visit(TupleAccessNode* node) { throw std::runtime_error("not implemented"); }
void MLIRGen::visit(TupleTypeCastNode* node) { throw std::runtime_error("not implemented"); }

void MLIRGen::visit(TypeCastNode* node) {
    node->expr->accept(*this);
    VarInfo from = popValue();
    VarInfo result = castType(&from, &node->type);
    pushValue(result);
}

void MLIRGen::visit(IdNode* node) {
    VarInfo* varInfo = currScope_->resolveVar(node->id);

    if (!varInfo || !varInfo->value) {
        throw SymbolError(1, "Semantic Analysis: Variable '" + node->id + "' not initialized.");
    }

    pushValue(*varInfo); 

}

void MLIRGen::visit(TrueNode* node) {
    auto boolType = builder_.getI1Type();

    CompleteType completeType = CompleteType(BaseType::BOOL);
    VarInfo varInfo = VarInfo(completeType);
    allocaLiteral(&varInfo);

    auto constTrue = builder_.create<mlir::arith::ConstantOp>(
        loc_, boolType, builder_.getIntegerAttr(boolType, 1)
    );
    builder_.create<mlir::memref::StoreOp>(loc_, constTrue, varInfo.value, mlir::ValueRange{});

    pushValue(varInfo);
}


void MLIRGen::visit(FalseNode* node) {
    CompleteType completeType = CompleteType(BaseType::BOOL);
    VarInfo varInfo = VarInfo(completeType);

    auto boolType = builder_.getI1Type();
    allocaLiteral(&varInfo);
    auto constFalse = builder_.create<mlir::arith::ConstantOp>(
        loc_, boolType, builder_.getIntegerAttr(boolType, 0)
    );
    builder_.create<mlir::memref::StoreOp>(loc_, constFalse, varInfo.value, mlir::ValueRange{});

    pushValue(varInfo);
}

void MLIRGen::visit(CharNode* node) {
    CompleteType completeType = CompleteType(BaseType::CHARACTER);
    VarInfo varInfo = VarInfo(completeType);

    auto charType = builder_.getI8Type();
    allocaLiteral(&varInfo);
    auto constChar = builder_.create<mlir::arith::ConstantOp>(
        loc_, charType, builder_.getIntegerAttr(charType, static_cast<int>(node->value))
    );
    builder_.create<mlir::memref::StoreOp>(loc_, constChar, varInfo.value, mlir::ValueRange{});

    pushValue(varInfo);
}

void MLIRGen::visit(IntNode* node) {
    CompleteType completeType = CompleteType(BaseType::INTEGER);
    VarInfo varInfo = VarInfo(completeType);

    auto intType = builder_.getI32Type();
    allocaLiteral(&varInfo);
    auto constInt = builder_.create<mlir::arith::ConstantOp>(
        loc_, intType, builder_.getIntegerAttr(intType, node->value)
    );
    builder_.create<mlir::memref::StoreOp>(loc_, constInt, varInfo.value, mlir::ValueRange{});

    pushValue(varInfo);
}

void MLIRGen::visit(RealNode* node) {
    CompleteType completeType = CompleteType(BaseType::REAL);
    VarInfo varInfo = VarInfo(completeType);

    auto realType = builder_.getF32Type();
    allocaLiteral(&varInfo);
    auto constReal = builder_.create<mlir::arith::ConstantOp>(
        loc_, realType, builder_.getFloatAttr(realType, node->value)
    );
    builder_.create<mlir::memref::StoreOp>(loc_, constReal, varInfo.value, mlir::ValueRange{});

    pushValue(varInfo);
}

void MLIRGen::visit(TupleLiteralNode* node) {
    VarInfo tupleVarInfo(node->type);
    allocaLiteral(&tupleVarInfo);

    if (tupleVarInfo.mlirSubtypes.size() != node->elements.size()) {
        throw std::runtime_error("FATAL: mismatched mlirSubtypes and node->elements sizes.");
    }

    for (size_t i = 0; i < node->elements.size(); ++i) {
        node->elements[i]->accept(*this);
        VarInfo elemVarInfo = popValue();

        VarInfo &target = tupleVarInfo.mlirSubtypes[i];

        mlir::Value loadedVal = builder_.create<mlir::memref::LoadOp>(
            loc_, elemVarInfo.value, mlir::ValueRange{}
        );

        builder_.create<mlir::memref::StoreOp>(
            loc_, loadedVal, target.value, mlir::ValueRange{}
        );
    }

    pushValue(tupleVarInfo);
}

void MLIRGen::allocaLiteral(VarInfo* varInfo) {
    varInfo->isConst = true;
    switch (varInfo->type.baseType) {
        case BaseType::BOOL:
            varInfo->value = builder_.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI1Type()));
            break;

        case (BaseType::CHARACTER):
            varInfo->value = builder_.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI8Type())
            );
            break;

        case (BaseType::INTEGER):
            varInfo->value = builder_.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI32Type())
            );
            break;

        case (BaseType::REAL):
            varInfo->value = builder_.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getF32Type())
            );
            break;

        case (BaseType::TUPLE):
            if (varInfo->type.subTypes.size() < 2) {
                throw SizeError(1, "Error: Tuple must have at least 2 elements.");
            }

            for (CompleteType& subtype: varInfo->type.subTypes) {
                VarInfo mlirSubtype = VarInfo(subtype);
                mlirSubtype.isConst = true; // Literals are always const

                // Copy over type info into VarInfo's subtypes
                varInfo->mlirSubtypes.emplace_back(
                    mlirSubtype
                );
                allocaLiteral(&varInfo->mlirSubtypes.back());
            }
            break;

        default:
            throw std::runtime_error("allocaLiteral FATAL: unsupported type " +
                                    std::to_string(static_cast<int>(varInfo->type.baseType)));
    }
}

/* TODO: implement tuple-tuple casting for TupleTypeCastNode handling */
VarInfo MLIRGen::castType(VarInfo* from, CompleteType* toType) {
    VarInfo to = VarInfo(*toType);
    allocaLiteral(&to); // Create new value container

    switch (from->type.baseType) {
        case (BaseType::BOOL):
        {
            mlir::Value boolVal = builder_.create<mlir::memref::LoadOp>(loc_, from->value, mlir::ValueRange{}); // Load value
            switch (toType->baseType) {
                case BaseType::BOOL:                    // Bool -> Bool
                    builder_.create<mlir::memref::StoreOp>(
                        loc_, boolVal, to.value, mlir::ValueRange{});
                    break;
       
                case BaseType::INTEGER:                 // Bool -> Int
                {
                    mlir::Value intVal = builder_.create<mlir::arith::ExtUIOp>(
                            loc_, builder_.getI32Type(), boolVal
                        );
                    builder_.create<mlir::memref::StoreOp>(loc_, intVal, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::CHARACTER:               // Bool -> Char
                {
                    mlir::Value charVal = builder_.create<mlir::arith::ExtUIOp>(
                        loc_, builder_.getI8Type(), boolVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, charVal, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::REAL:                    // Bool -> Real
                {
                    mlir::Value realVal = builder_.create<mlir::arith::UIToFPOp>(
                        loc_, builder_.getF32Type(), boolVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, realVal, to.value, mlir::ValueRange{});
                    break;
                }

                default:
                    throw LiteralError(1, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            break;
        }

        case (BaseType::CHARACTER):
        {
            mlir::Value chVal = builder_.create<mlir::memref::LoadOp>(loc_, from->value, mlir::ValueRange{});
            switch (toType->baseType) {
                case BaseType::CHARACTER:               // Char -> Char
                    builder_.create<mlir::memref::StoreOp>(loc_, chVal, to.value, mlir::ValueRange{});
                    break;

                case BaseType::BOOL:                    // Char -> Bool
                {
                    mlir::Value zeroConst = builder_.create<mlir::arith::ConstantOp>(
                        loc_, builder_.getI8Type(), builder_.getIntegerAttr(builder_.getI8Type(), 0)
                    );
                    mlir::Value isZeroConst = builder_.create<mlir::arith::CmpIOp>(
                        loc_, mlir::arith::CmpIPredicate::ne, chVal, zeroConst
                    );  // '\0' == false
                    builder_.create<mlir::memref::StoreOp>(loc_, isZeroConst, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::INTEGER:                 // Char -> Int
                {
                    mlir::Value intVal = builder_.create<mlir::arith::ExtUIOp>(
                            loc_, builder_.getI32Type(), chVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, intVal, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::REAL:                    // Char -> Real
                {
                    mlir::Value realVal = builder_.create<mlir::arith::UIToFPOp>(
                            loc_, builder_.getF32Type(), chVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, realVal, to.value, mlir::ValueRange{});
                    break;
                }

                default:
                    throw LiteralError(1, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            break;
        }

        case (BaseType::INTEGER):
        {
            mlir::Value i32Val = builder_.create<mlir::memref::LoadOp>(loc_, from->value, mlir::ValueRange{});
            switch (toType->baseType) {
                case BaseType::INTEGER:                 // Int -> Int
                    builder_.create<mlir::memref::StoreOp>(loc_, i32Val, to.value, mlir::ValueRange{});
                    break;

                case BaseType::BOOL:                    // Int -> Bool (ne 0)
                {
                    mlir::Value zero = builder_.create<mlir::arith::ConstantOp>(
                        loc_, builder_.getI32Type(), builder_.getIntegerAttr(builder_.getI32Type(), 0)
                    );
                    mlir::Value neZeroConstant = builder_.create<mlir::arith::CmpIOp>(
                        loc_, mlir::arith::CmpIPredicate::ne, i32Val, zero
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, neZeroConstant, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::CHARACTER:               // Int -> Char (mod 256)
                {
                    mlir::Value i8Val = builder_.create<mlir::arith::TruncIOp>(
                        loc_, builder_.getI8Type(), i32Val
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, i8Val, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::REAL:                    // Int -> Real
                {
                    mlir::Value fVal = builder_.create<mlir::arith::SIToFPOp>(
                        loc_, builder_.getF32Type(), i32Val
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, fVal, to.value, mlir::ValueRange{});
                    break;
                }

                default:
                    throw LiteralError(1, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            break;
        }

        case (BaseType::REAL):
        {
            mlir::Value fVal = builder_.create<mlir::memref::LoadOp>(loc_, from->value, mlir::ValueRange{});
            switch (toType->baseType) {
                case BaseType::REAL:                    // Real -> Real
                    builder_.create<mlir::memref::StoreOp>(loc_, fVal, to.value, mlir::ValueRange{});
                    break;

                case BaseType::INTEGER:                 // Real -> Int (truncate)
                {
                    mlir::Value iVal = builder_.create<mlir::arith::FPToSIOp>(
                        loc_, builder_.getI32Type(), fVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, iVal, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::CHARACTER:               // Real -> Char (not allowed)
                case BaseType::BOOL:                    // Real -> Bool (not allowed)
                default:
                    throw LiteralError(1, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            break;
        }

        default:
            throw LiteralError(1, std::string("Codegen: unsupported cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
    }

    return to;
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

void MLIRGen::visit(ExpExpr* node) {
    node->left->accept(*this);
    mlir::Value lhs = popValue();
    node->right->accept(*this);
    mlir::Value rhs = popValue();

    bool isInt = lhs.getType().isa<mlir::IntegerType>();

    // Promote to float if needed
    if (isInt) {
        auto f32Type = builder_.getF32Type();
        lhs = builder_.create<mlir::arith::SIToFPOp>(loc_, f32Type, lhs);
        rhs = builder_.create<mlir::arith::SIToFPOp>(loc_, f32Type, rhs);
    }

    mlir::Value result = builder_.create<mlir::math::PowFOp>(loc_, lhs, rhs);

    // If original operands were int, apply math.floor and cast back to int
    if (isInt) {
        mlir::Value floored = builder_.create<mlir::math::FloorOp>(loc_, result);
        auto intType = builder_.getI32Type();
        result = builder_.create<mlir::arith::FPToSIOp>(loc_, intType, floored);
    }

    pushValue(result);
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

void MLIRGen::visit(CompExpr* node) {
    node->left->accept(*this);
    mlir::Value left = popValue();
    node->right->accept(*this);
    mlir::Value right = popValue();

    mlir::Value cmp;
    if (left.getType().isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate predicate;
        switch (node->op) {
            case '<':
                predicate = mlir::arith::CmpIPredicate::slt;
                break;
            case '<=':
                predicate = mlir::arith::CmpIPredicate::sle;
                break;
            case '>':
                predicate = mlir::arith::CmpIPredicate::sgt;
                break;
            case '>=':
                predicate = mlir::arith::CmpIPredicate::sge;
                break;
            default:
                throw std::runtime_error("MLIRGen Error: Unsupported comparison operator for integers.");
        }
        cmp = builder_.create<mlir::arith::CmpIOp>(loc_, predicate, left, right);
    } else if (left.getType().isa<mlir::FloatType>()) {
        mlir::arith::CmpFPredicate predicate;
        switch (node->op) {
            case '<':
                predicate = mlir::arith::CmpFPredicate::OLT;
                break;
            case '<=':
                predicate = mlir::arith::CmpFPredicate::OLE;
                break;
            case '>':
                predicate = mlir::arith::CmpFPredicate::OGT;
                break;
            case '>=':
                predicate = mlir::arith::CmpFPredicate::OGE;
                break;
            default:
                throw std::runtime_error("MLIRGen Error: Unsupported comparison operator for reals.");
        }
        cmp = builder_.create<mlir::arith::CmpFOp>(loc_, predicate, left, right);
    } else {
        throw std::runtime_error("MLIRGen Error: Unsupported type for comparison.");
    }
    pushValue(cmp);
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