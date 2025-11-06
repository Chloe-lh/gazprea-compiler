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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
void MLIRGen::visit(FuncStatNode* node) { throw std::runtime_error("FuncStatNode not implemented"); }
void MLIRGen::visit(FuncPrototypeNode* node) { throw std::runtime_error("FuncPrototypeNode not implemented"); }
void MLIRGen::visit(FuncBlockNode* node) { throw std::runtime_error("FuncBlockNode not implemented"); }
void MLIRGen::visit(ProcedureNode* node) { throw std::runtime_error("ProcedureNode not implemented"); }

// Declarations
void MLIRGen::visit(TypedDecNode* node) { throw std::runtime_error("TypedDecNode not implemented"); }
void MLIRGen::visit(InferredDecNode* node) { throw std::runtime_error("InferredDecNode not implemented"); }
void MLIRGen::visit(TupleTypedDecNode* node) { throw std::runtime_error("TupleTypedDecNode not implemented"); }
void MLIRGen::visit(TypeAliasDecNode* node) { throw std::runtime_error("TypeAliasDecNode not implemented"); }
void MLIRGen::visit(TypeAliasNode* node) { throw std::runtime_error("TypeAliasNode not implemented"); }
void MLIRGen::visit(TupleTypeAliasNode* node) { throw std::runtime_error("TupleTypeAliasNode not implemented"); }

// Statements
void MLIRGen::visit(AssignStatNode* node) { throw std::runtime_error("AssignStatNode not implemented"); }
void MLIRGen::visit(OutputStatNode* node) { throw std::runtime_error("OutputStatNode not implemented"); }
void MLIRGen::visit(InputStatNode* node) { throw std::runtime_error("InputStatNode not implemented"); }
void MLIRGen::visit(BreakStatNode* node) { throw std::runtime_error("BreakStatNode not implemented"); }
void MLIRGen::visit(ContinueStatNode* node) { throw std::runtime_error("ContinueStatNode not implemented"); }
void MLIRGen::visit(ReturnStatNode* node) { throw std::runtime_error("ReturnStatNode not implemented"); }
void MLIRGen::visit(CallStatNode* node) { throw std::runtime_error("CallStatNode not implemented"); }
void MLIRGen::visit(IfNode* node) {
    // Evaluate the condition expression
    node->cond->accept(*this);
    VarInfo condVarInfo = popValue();
    
    // Load the boolean value from the condition
    mlir::Value conditionValue = builder_.create<mlir::memref::LoadOp>(
        loc_, condVarInfo.value, mlir::ValueRange{}
    );
    
    // Determine if we have an else branch
    bool hasElse = (node->elseBlock != nullptr) || (node->elseStat != nullptr);
    
    // Create the scf.if operation
    auto ifOp = builder_.create<mlir::scf::IfOp>(
        loc_, conditionValue, hasElse
    );
    
    // Build the 'then' region
    builder_.setInsertionPointToStart(&ifOp.getThenRegion().front());
    if (node->thenBlock) {
        node->thenBlock->accept(*this);
    } else if (node->thenStat) {
        node->thenStat->accept(*this);
    }
    builder_.create<mlir::scf::YieldOp>(loc_);
    
    // Build the 'else' region if it exists
    if (hasElse) {
        builder_.setInsertionPointToStart(&ifOp.getElseRegion().front());
        if (node->elseBlock) {
            node->elseBlock->accept(*this);
        } else if (node->elseStat) {
            node->elseStat->accept(*this);
        }
        builder_.create<mlir::scf::YieldOp>(loc_);
    }
    
    // Restore insertion point after the if operation
    builder_.setInsertionPointAfter(ifOp);
}
void MLIRGen::visit(LoopNode* node) { throw std::runtime_error("LoopNode not implemented"); }
void MLIRGen::visit(BlockNode* node) { 
    throw std::runtime_error("BlockNode not implemented"); 
}

// Expressions / Operators
void MLIRGen::visit(ParenExpr* node) { node->expr->accept(*this); }
void MLIRGen::visit(FuncCallExpr* node) { throw std::runtime_error("FuncCallExpr not implemented");}
void MLIRGen::visit(UnaryExpr* node) { throw std::runtime_error("UnaryExpr not implemented"); }
void MLIRGen::visit(ExpExpr* node) { throw std::runtime_error("ExpExpr not implemented"); }
void MLIRGen::visit(MultExpr* node) { throw std::runtime_error("MultExpr not implemented"); }
void MLIRGen::visit(AddExpr* node) { throw std::runtime_error("AddExpr not implemented"); }
void MLIRGen::visit(CompExpr* node) {
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
void MLIRGen::visit(NotExpr* node) { throw std::runtime_error("NotExpr not implemented"); }
void MLIRGen::visit(EqExpr* node) { throw std::runtime_error("EqExpr not implemented"); }
void MLIRGen::visit(AndExpr* node) { throw std::runtime_error("AndExpr not implemented"); }
void MLIRGen::visit(OrExpr* node) { throw std::runtime_error("OrExpr not implemented"); }
void MLIRGen::visit(TupleAccessNode* node) { throw std::runtime_error("TupleAccessNode not implemented"); }
void MLIRGen::visit(TupleTypeCastNode* node) { throw std::runtime_error("TupleTypeCastNode not implemented"); }

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
