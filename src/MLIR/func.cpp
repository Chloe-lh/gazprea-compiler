#include "CompileTimeExceptions.h"
#include "MLIRgen.h"

void MLIRGen::visit(FuncStatNode* node) {
    Scope* savedScope = nullptr;
    beginFunctionDefinitionWithConstants(
        node, node->name, node->parameters, node->returnType, savedScope);

    // Handle FuncStat nodes since they have no body
    if (node->returnStat) {
        node->returnStat->accept(*this);
    }

    // Handle block funcs/procedures
    lowerFunctionOrProcedureBody(
        node->parameters, node->body, node->returnType, savedScope);
}

void MLIRGen::visit(FuncBlockNode* node) {
    Scope* savedScope = nullptr;
    beginFunctionDefinitionWithConstants(node, node->name, node->parameters, node->returnType, savedScope);
    lowerFunctionOrProcedureBody(node->parameters, node->body, node->returnType, savedScope);
}

void MLIRGen::visit(FuncPrototypeNode* node) {
    std::vector<mlir::Type> argTys, resTys;
    argTys.reserve(node->parameters.size());
    for (const auto& p : node->parameters) argTys.push_back(getLLVMType(p.type));
    if (node->returnType.baseType != BaseType::UNKNOWN)
        resTys.push_back(getLLVMType(node->returnType));

    auto ftype = builder_.getFunctionType(argTys, resTys);
    if (!module_.lookupSymbol<mlir::func::FuncOp>(node->name)) {
        builder_.create<mlir::func::FuncOp>(loc_, node->name, ftype);
    }
}
void MLIRGen::visit(ProcedurePrototypeNode* node) {
    // Procedures and functions share the same lowering convention in MLIR:
    // they both become `func.func` with (possibly empty) result type.
    std::vector<mlir::Type> argTys, resTys;
    argTys.reserve(node->params.size());
    for (const auto& p : node->params) {
        argTys.push_back(getLLVMType(p.type));
    }
    if (node->returnType.baseType != BaseType::UNKNOWN) {
        resTys.push_back(getLLVMType(node->returnType));
    }

    auto ftype = builder_.getFunctionType(argTys, resTys);
    if (!module_.lookupSymbol<mlir::func::FuncOp>(node->name)) {
        builder_.create<mlir::func::FuncOp>(loc_, node->name, ftype);
    }
}
void MLIRGen::visit(ProcedureBlockNode* node) {
    Scope* savedScope = nullptr;
    beginFunctionDefinitionWithConstants(node, node->name, node->params, node->returnType, savedScope);
    lowerFunctionOrProcedureBody(node->params, node->body, node->returnType, savedScope);
}

/*
Handles both functions AND procedures when called as an expression.
*/
void MLIRGen::visit(FuncCallExpr* node) {
    if (!currScope_) {
        throw std::runtime_error("FuncCallExpr: no current scope");
    }
    if (!node) {
        throw std::runtime_error("FuncCallExpr: null node");
    }

    // Evaluate argument expressions and collect their VarInfo
    std::vector<VarInfo> argInfos;
    argInfos.reserve(node->args.size());
    for (const auto &argExpr : node->args) {
        if (!argExpr) {
            throw std::runtime_error("FuncCallExpr: null argument expression");
        }
        argExpr->accept(*this);
        if (v_stack_.empty()) {
            throw std::runtime_error("FuncCallExpr: argument evaluation did not produce a value");
        }
        VarInfo argInfo = popValue();
        argInfos.push_back(argInfo);
    }

    // Build type-only VarInfo list for callee resolution
    std::vector<VarInfo> typeOnlyArgs;
    typeOnlyArgs.reserve(argInfos.size());
    for (const auto &argInfo : argInfos) {
        typeOnlyArgs.emplace_back(VarInfo{"", argInfo.type, true});
    }

    // Try resolve as function first
    FuncInfo *funcInfo = nullptr;
    ProcInfo *procInfo = nullptr;

    try {
        funcInfo = currScope_->resolveFunc(node->funcName, typeOnlyArgs);
    } catch (const CompileTimeException &) {
        funcInfo = nullptr;
    }

    bool isFunction = funcInfo != nullptr;

    if (!isFunction) {
        // Not a function; try resolving as a procedure used as expression.
        try {
            procInfo = currScope_->resolveProc(node->funcName, typeOnlyArgs);
        } catch (const CompileTimeException &) {
            procInfo = nullptr;
        }

        if (!procInfo) {
            throw std::runtime_error("FuncCallExpr: callee '" + node->funcName + "' not found as function or procedure during codegen");
        }
        // safety check - should not happen
        if (procInfo->procReturn.baseType == BaseType::UNKNOWN) {
            throw std::runtime_error("FuncCallExpr: procedure '" + node->funcName + "' used as expression but has no return type (codegen)");
        }
    } else {
        // For functions, semantic analysis ensures a concrete return type.
        if (funcInfo->funcReturn.baseType == BaseType::UNKNOWN) {
            throw std::runtime_error("FuncCallExpr: function '" + node->funcName + "' has UNKNOWN return type in codegen");
        }
    }

    // Look up the callee function op in the module
    mlir::func::FuncOp calleeFunc =
        module_.lookupSymbol<mlir::func::FuncOp>(node->funcName);
    if (!calleeFunc) {
        throw std::runtime_error("FuncCallExpr: callee function '" + node->funcName + "' not found in module");
    }

    // Build argument values for MLIR call
    std::vector<mlir::Value> callArgs;
    callArgs.reserve(argInfos.size());

    const auto &paramInfos =
        isFunction ? funcInfo->params : procInfo->params;

    for (size_t i = 0; i < argInfos.size() && i < paramInfos.size(); ++i) {
        const auto &param = paramInfos[i];
        const auto &argInfo = argInfos[i];

        if (param.type.baseType == BaseType::TUPLE) {
            // tuple uses ptr to llvm struct repr
            if (!argInfo.value) {
                throw std::runtime_error(
                    "FuncCallExpr: tuple argument has no value");
            }
            if (!param.isConst) {           // var tuple, pass ptr in
                callArgs.push_back(argInfo.value);
            } else {
                // const tuple, pass by value
                mlir::Type structTy = getLLVMType(param.type);
                mlir::Value structVal = builder_.create<mlir::LLVM::LoadOp>(
                    loc_, structTy, argInfo.value);
                callArgs.push_back(structVal);
            }
        } else {
            if (!param.isConst) {
                // Scalar var parameter: pass memref directly.
                if (!argInfo.value) {
                    throw std::runtime_error(
                        "FuncCallExpr: var parameter requires mutable argument (variable), but argument has no value");
                }
                mlir::Type argType = argInfo.value.getType();
                if (!argType.isa<mlir::MemRefType>()) {
                    throw std::runtime_error(
                        "FuncCallExpr: var parameter requires mutable argument (variable) with memref type");
                }
                callArgs.push_back(argInfo.value);
            } else {
                // Scalar const parameter: pass loaded value.
                if (!argInfo.value) {
                    throw std::runtime_error(
                        "FuncCallExpr: argument has no value");
                }
                mlir::Value argVal;
                // Normalize to an SSA value (getSSAValue will load memref if needed)
                argVal = getSSAValue(argInfo);
                callArgs.push_back(argVal);
            }
        }
    }

    auto funcType = calleeFunc.getFunctionType();
    if (funcType.getNumInputs() != callArgs.size()) {
        throw std::runtime_error("FuncCallExpr: argument count mismatch for callee '" + node->funcName + "'");
    }

    // Verify each argument type matches the function signature
    for (size_t i = 0; i < callArgs.size(); ++i) {
        mlir::Type expectedType = funcType.getInput(i);
        mlir::Type actualType = callArgs[i].getType();
        if (expectedType != actualType) {
            throw std::runtime_error("FuncCallExpr: type mismatch for argument " + std::to_string(i) + " in call to '" + node->funcName + "'");
        }
    }

    if (!builder_.getBlock()) {
        throw std::runtime_error("FuncCallExpr: builder has no current block");
    }

    auto callOp =
        builder_.create<mlir::func::CallOp>(loc_, calleeFunc, callArgs);

    // Handle return value (must exist for expression use)
    if (funcType.getNumResults() == 0) {
        throw std::runtime_error("FuncCallExpr: callee '" + node->funcName +
                                 "' used as expression but has no return value");
    }
    if (funcType.getNumResults() != 1) {
        throw std::runtime_error("FuncCallExpr: multiple return values not supported for callee '" +
                                 node->funcName + "'");
    }

    mlir::Value retVal = callOp.getResult(0);

    // Wrap returned value in a VarInfo with appropriate storage.
    VarInfo resultVar(node->type);
    allocaLiteral(&resultVar);
    if (node->type.baseType == BaseType::TUPLE) {
        // For tuple returns, resultVar.value is an !llvm.ptr; store the
        // returned struct into it.
        builder_.create<mlir::LLVM::StoreOp>(loc_, retVal, resultVar.value);
    } else {
        builder_.create<mlir::memref::StoreOp>(loc_, retVal, resultVar.value,
                                               mlir::ValueRange{});
    }

    pushValue(resultVar);
}
