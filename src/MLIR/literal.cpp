#include "CompileTimeExceptions.h"
#include "MLIRgen.h"
#include "Scope.h"
#include "Types.h"


void MLIRGen::visit(IdNode* node) {
    if (!currScope_) {
        throw std::runtime_error("visit(IdNode*): no current scope");
    }

    // Prefer the binding established during semantic analysis to honour
    // declaration order and shadowing (e.g., parameter vs local with same name).
    VarInfo* varInfo = node->binding;
    if (!varInfo) {
        throw std::runtime_error("visit(IdNode*): IdNode did not receive bound variable");
    }

    if (!varInfo) {
        throw SymbolError(node->line, "Semantic Analysis: Variable '" + node->id + "' not defined.");
    }

    // If the variable doesn't have a value, it's likely a global variable
    // (or an uninitialised local); we may need to allocate storage and
    // initialise from globals.
    bool needsAllocation = !varInfo->value;
    
    if (needsAllocation) {
        if (varInfo->type.baseType == BaseType::TUPLE ||
            varInfo->type.baseType == BaseType::STRUCT) {
            // Tuple/struct aggregates: try to resolve as a global first;
            // otherwise, allocate local storage.
            auto globalOp =
                module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->id);
            if (globalOp) {
                varInfo->value =
                    builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
            } else {
                allocaVar(varInfo, node->line);
                if (!varInfo->value) {
                    throw std::runtime_error(
                        "visit(IdNode*): Failed to allocate aggregate variable '" +
                        node->id + "'");
                }
            }
        } else if (varInfo->type.baseType == BaseType::ARRAY) {
            // TODO: Implement global arrays
            allocaVar(varInfo, node->line);
            if (!varInfo->value) {
                throw std::runtime_error(
                    "visit(IdNode*): Failed to allocate array variable '" +
                    node->id + "'");
            }
        } else if (isScalarType(varInfo->type.baseType)) {
            // Scalar: try to resolve as a global first
            auto globalOp = module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->id);
            if (!globalOp) {
                // Not a global - ensure local variable is allocated
                allocaVar(varInfo, node->line);
                if (!varInfo->value) {
                    throw std::runtime_error("visit(IdNode*): Failed to allocate variable '" + node->id + "'");
                }
            } else {
                // Scalar global: load into a temporary VarInfo so downstream
                // code can keep using memref-based paths.
                mlir::Value globalAddr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
                mlir::Type elementType = globalOp.getType();

                VarInfo tempVarInfo = VarInfo(varInfo->type);
                if (tempVarInfo.type.baseType == BaseType::UNKNOWN) {
                    throw std::runtime_error("visit(IdNode*): Variable '" + node->id + "' has UNKNOWN type");
                }
                allocaLiteral(&tempVarInfo, node->line);

                mlir::Value loadedValue = builder_.create<mlir::LLVM::LoadOp>(
                    loc_, elementType, globalAddr);
                builder_.create<mlir::memref::StoreOp>(loc_, loadedValue, tempVarInfo.value, mlir::ValueRange{});

                pushValue(tempVarInfo);
                return;
            }
        } else {
            throw std::runtime_error("MLIRGen::IdNode: Unsupported type in alloc");
        }
    }

    // Ensure we have a valid value before pushing
    if (!varInfo->value) {
        throw std::runtime_error("visit(IdNode*): Variable '" + node->id +
                                 "' has no value after allocation");
    }
    
    pushValue(*varInfo); 
}

void MLIRGen::visit(TrueNode* node) {
    auto boolType = builder_.getI1Type();
    auto c = builder_.create<mlir::arith::ConstantOp>(loc_, boolType,
                builder_.getIntegerAttr(boolType, 1)); // 1 == true
    VarInfo v = (CompleteType(BaseType::BOOL));
    v.value = c.getResult();
    v.isLValue = false;
    pushValue(v);
}


void MLIRGen::visit(FalseNode* node) {
    auto boolType = builder_.getI1Type();
    auto c = builder_.create<mlir::arith::ConstantOp>(loc_, boolType,
                builder_.getIntegerAttr(boolType, 0)); // 0 == false
    VarInfo v = (CompleteType(BaseType::BOOL));
    v.value = c.getResult();
    v.isLValue = false;
    pushValue(v);
}


void MLIRGen::visit(CharNode* node) {
    // Emit character literals as SSA constants (i8) rather than allocating
    // a temporary memref and storing into it. This yields cleaner, more
    // efficient MLIR where temporaries are SSA values.
    CompleteType completeType = CompleteType(BaseType::CHARACTER);
    VarInfo v = VarInfo(completeType);

    auto charType = builder_.getI8Type();
    auto constChar = builder_.create<mlir::arith::ConstantOp>(
        loc_, charType, builder_.getIntegerAttr(charType, static_cast<int>(node->value))
    );
    v.value = constChar.getResult();
    v.isLValue = false;
    pushValue(v);
}

void MLIRGen::visit(IntNode* node) {
    // Emit integer literals as SSA constants (i32) instead of temporaries.
    CompleteType completeType = CompleteType(BaseType::INTEGER);
    VarInfo v = VarInfo(completeType);

    auto intType = builder_.getI32Type();
    auto constInt = builder_.create<mlir::arith::ConstantOp>(
        loc_, intType, builder_.getIntegerAttr(intType, node->value)
    );
    v.value = constInt.getResult();
    v.isLValue = false;
    pushValue(v);
}

void MLIRGen::visit(RealNode* node) {
    // Emit real (float) literals as SSA constants (f32) instead of temporaries.
    CompleteType completeType = CompleteType(BaseType::REAL);
    VarInfo v = VarInfo(completeType);

    auto realType = builder_.getF32Type();
    auto constReal = builder_.create<mlir::arith::ConstantOp>(
        loc_, realType, builder_.getFloatAttr(realType, node->value)
    );
    v.value = constReal.getResult();
    v.isLValue = false;
    pushValue(v);
}

void MLIRGen::visit(StringNode* node) {
    throw std::runtime_error("String expressions not supported outside of output statements yet.");
}
