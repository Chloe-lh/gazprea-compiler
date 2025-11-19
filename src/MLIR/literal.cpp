#include "CompileTimeExceptions.h"
#include "MLIRgen.h"


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
        throw SymbolError(1, "Semantic Analysis: Variable '" + node->id + "' not defined.");
    }

    // If the variable doesn't have a value, it's likely a global variable
    // (or an uninitialised local); we may need to allocate storage and
    // initialise from globals.
    bool needsAllocation = !varInfo->value;
    
    if (needsAllocation) {
        if (varInfo->type.baseType == BaseType::TUPLE) {
            // Tuple: try to resolve as a global struct first; otherwise,allocate local storage.
            auto globalOp =
                module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->id);
            if (globalOp) {
                varInfo->value =
                    builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
            } else {
                allocaVar(varInfo);
                if (!varInfo->value) {
                    throw std::runtime_error(
                        "visit(IdNode*): Failed to allocate tuple variable '" +
                        node->id + "'");
                }
            }
        } else {
            // Scalar: try to resolve as a global first
            auto globalOp = module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->id);
            if (!globalOp) {
                // Not a global - ensure local variable is allocated
                allocaVar(varInfo);
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
                allocaLiteral(&tempVarInfo);

                mlir::Value loadedValue = builder_.create<mlir::LLVM::LoadOp>(
                    loc_, elementType, globalAddr);
                builder_.create<mlir::memref::StoreOp>(loc_, loadedValue, tempVarInfo.value, mlir::ValueRange{});

                pushValue(tempVarInfo);
                return;
            }
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

void MLIRGen::visit(StringNode* node) {
    throw std::runtime_error("String expressions not supported outside of output statements yet.");
}

