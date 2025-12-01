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
Handles functions, procedures AND struct literals when called as an expression.
Dispatches based on callType determined by Semantic Analysis.
*/
void MLIRGen::visit(FuncCallExprOrStructLiteral* node) {
    if (!currScope_) {
        throw std::runtime_error("FuncCallExprOrStructLiteral: no current scope");
    }
    if (!node) {
        throw std::runtime_error("FuncCallExprOrStructLiteral: null node");
    }

    if (node->callType == CallType::STRUCT_LITERAL) {
        // --- Struct Literal Construction ---
        
        VarInfo structVar(node->type);
        allocaLiteral(&structVar, node->line);

        if (!structVar.value) {
            throw std::runtime_error("FuncCallExprOrStructLiteral: failed to allocate struct storage.");
        }

        // Lower to struct type
        mlir::Type structTy = getLLVMType(node->type);
        mlir::Value structVal = builder_.create<mlir::LLVM::UndefOp>(loc_, structTy);

        // Evaluate each argument and insert into the struct
        if (node->args.size() != node->type.subTypes.size()) {
             throw std::runtime_error("FuncCallExprOrStructLiteral: struct argument count mismatch.");
        }

        for (size_t i = 0; i < node->args.size(); ++i) {
            if (!node->args[i]) throw std::runtime_error("FuncCallExprOrStructLiteral: null argument");
            
            node->args[i]->accept(*this);
            VarInfo argInfo = popValue();

            // Ensure the argument matches the field type (promotions, etc.)
            CompleteType fieldType = node->type.subTypes[i];
            VarInfo promoted = promoteType(&argInfo, &fieldType, node->line);
            mlir::Value loadedVal = getSSAValue(promoted);

            llvm::SmallVector<int64_t, 1> pos{static_cast<int64_t>(i)};
            structVal = builder_.create<mlir::LLVM::InsertValueOp>(
                loc_, structVal, loadedVal, pos);
        }

        // Store and return struct
        builder_.create<mlir::LLVM::StoreOp>(loc_, structVal, structVar.value);
        pushValue(structVar);
        return;
    }

    // ---------- Now handle Function or procedure call
    
    // Identify Callee Info
    std::vector<VarInfo> paramInfos;
    bool isFunction = (node->callType == CallType::FUNCTION);

    if (isFunction) {
        if (node->resolvedFunc) {
            // Use cached info from semantic analysis if available
            paramInfos = node->resolvedFunc->params;
        } 
        // If not cached, we will resolve below using arg types
    }

    // 2. Evaluate Arguments
    std::vector<VarInfo> argInfos;
    argInfos.reserve(node->args.size());
    for (const auto &argExpr : node->args) {
        if (!argExpr) throw std::runtime_error("FuncCallExprOrStructLiteral: null argument expression");
        argExpr->accept(*this);
        if (v_stack_.empty()) throw std::runtime_error("FuncCallExprOrStructLiteral: argument evaluation did not produce a value");
        argInfos.push_back(popValue());
    }

    // Resolve Callee, get paramInfo
    if (paramInfos.empty()) {
        // Build type-only VarInfo list for resolution
        std::vector<VarInfo> typeOnlyArgs;
        typeOnlyArgs.reserve(argInfos.size());
        for (const auto &argInfo : argInfos) {
            typeOnlyArgs.emplace_back(VarInfo{"", argInfo.type, true});
        }

        if (isFunction) {
            try {
                FuncInfo* fi = currScope_->resolveFunc(node->funcName, typeOnlyArgs, node->line);
                paramInfos = fi->params;
            } catch (...) {
                throw std::runtime_error("FuncCallExprOrStructLiteral: could not resolve function '" + node->funcName + "' during codegen");
            }
        } else {
            // Handle procedure
            try {
                ProcInfo* pi = currScope_->resolveProc(node->funcName, typeOnlyArgs, node->line);
                paramInfos = pi->params;
                if (pi->procReturn.baseType == BaseType::UNKNOWN) {
                     throw std::runtime_error("FuncCallExprOrStructLiteral: procedure '" + node->funcName + "' used as expression has no return type");
                }
            } catch (...) {
                throw std::runtime_error("FuncCallExprOrStructLiteral: could not resolve procedure '" + node->funcName + "' during codegen");
            }
        }
    }

    // Build MLIR Call args
    mlir::func::FuncOp calleeFunc = module_.lookupSymbol<mlir::func::FuncOp>(node->funcName);
    if (!calleeFunc) {
        throw std::runtime_error("FuncCallExprOrStructLiteral: callee function '" + node->funcName + "' not found in module");
    }

    std::vector<mlir::Value> callArgs;
    callArgs.reserve(argInfos.size());

    if (argInfos.size() != paramInfos.size()) {
         throw std::runtime_error("FuncCallExprOrStructLiteral: argument count mismatch.");
    }

    for (size_t i = 0; i < argInfos.size(); ++i) {
        const auto &param = paramInfos[i];
        const auto &argInfo = argInfos[i];

        if (param.type.baseType == BaseType::TUPLE) {
            if (!argInfo.value) throw std::runtime_error("FuncCallExprOrStructLiteral: tuple argument has no value");
            
            if (!param.isConst) {
                // var tuple: pass ptr directly
                callArgs.push_back(argInfo.value);
            } else {
                // const tuple: pass by value (load from ptr)
                mlir::Type structTy = getLLVMType(param.type);
                mlir::Value structVal = builder_.create<mlir::LLVM::LoadOp>(loc_, structTy, argInfo.value);
                callArgs.push_back(structVal);
            }
        } else {
            if (!param.isConst) {
                // Scalar var parameter: pass memref directly.
                if (!argInfo.value) throw std::runtime_error("FuncCallExprOrStructLiteral: var param requires value");
                if (!argInfo.value.getType().isa<mlir::MemRefType>()) {
                     throw std::runtime_error("FuncCallExprOrStructLiteral: var param requires memref");
                }
                callArgs.push_back(argInfo.value);
            } else {
                // Const parameter: implicit promotion + pass SSA value
                if (!argInfo.value) throw std::runtime_error("FuncCallExprOrStructLiteral: argument has no value");
                
                VarInfo argCopy = argInfo; 
                CompleteType targetType = param.type;
                VarInfo promoted = promoteType(&argCopy, &targetType, node->line);
                
                callArgs.push_back(getSSAValue(promoted));
            }
        }
    }

    // Generate call op
    if (!builder_.getBlock()) throw std::runtime_error("FuncCallExprOrStructLiteral: builder has no current block");

    auto callOp = builder_.create<mlir::func::CallOp>(loc_, calleeFunc, callArgs);

    // Handle return val
    auto funcType = calleeFunc.getFunctionType();
    if (funcType.getNumResults() != 1) {
        throw std::runtime_error("FuncCallExprOrStructLiteral: callee '" + node->funcName + "' must return exactly one value");
    }

    mlir::Value retVal = callOp.getResult(0);

    // Wrap returned value in a VarInfo 
    VarInfo resultVar(node->type);
    allocaLiteral(&resultVar, node->line);
    if (node->type.baseType == BaseType::TUPLE) {
        builder_.create<mlir::LLVM::StoreOp>(loc_, retVal, resultVar.value);
    } else {
        builder_.create<mlir::memref::StoreOp>(loc_, retVal, resultVar.value, mlir::ValueRange{});
    }

    pushValue(resultVar);
}