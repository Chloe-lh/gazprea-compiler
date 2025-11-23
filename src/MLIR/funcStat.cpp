#include "CompileTimeExceptions.h"
#include "MLIRgen.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

void MLIRGen::visit(BreakStatNode* node) {
    if (loopContexts_.empty()) {
        throw std::runtime_error("BreakStatNode: break statement outside of loop");
    }
    auto& loopCtx = loopContexts_.back();
    if (!loopCtx.exitBlock) {
        throw std::runtime_error("BreakStatNode: loop context missing exit block");
    }
    // Branch to the loop's exit block
    builder_.create<mlir::cf::BranchOp>(loc_, loopCtx.exitBlock);
}

void MLIRGen::visit(ContinueStatNode* node) {
    if (loopContexts_.empty()) {
        throw std::runtime_error("ContinueStatNode: continue statement outside of loop");
    }
    auto& loopCtx = loopContexts_.back();
    if (!loopCtx.continueBlock) {
        throw std::runtime_error("ContinueStatNode: loop context missing continue block");
    }
    // Branch to the loop's continue block (typically the condition block or
    // directly to the loop body for plain loops).
    builder_.create<mlir::cf::BranchOp>(loc_, loopCtx.continueBlock);
}

void MLIRGen::visit(ReturnStatNode* node) {
    const CompleteType* retTy = currScope_ ? currScope_->getReturnType() : nullptr;
    if (!retTy) {
        throw std::runtime_error("visit(ReturnStatNode*): no enclosing function");
    }

    // Void return: emit direct func.return with no operands.
    if (retTy->baseType == BaseType::UNKNOWN) {
        builder_.create<mlir::func::ReturnOp>(loc_);
        return;
    }
    if (!node->expr) {
        throw std::runtime_error("Codegen: missing return value for non-void procedure/function");
    }
    node->expr->accept(*this);
    VarInfo v = popValue();

    if (v.type.baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("visit(ReturnStatNode*): return expression has UNKNOWN type");
    }
    VarInfo promoted = promoteType(&v, const_cast<CompleteType*>(retTy), node->line);
    mlir::Value retVal;
    if (retTy->baseType == BaseType::TUPLE) {
        mlir::Type structTy = getLLVMType(*retTy);
        retVal = builder_.create<mlir::LLVM::LoadOp>(
            loc_, structTy, promoted.value);
    } else {
        // Normalize scalar return value to SSA (load memref if needed)
        retVal = getSSAValue(promoted);
    }
    builder_.create<mlir::func::ReturnOp>(loc_, retVal);
}

void MLIRGen::visit(CallStatNode* node) {
    if (!node || !node->call) {
        throw std::runtime_error("CallStatNode: missing call expression");
    }
    
    if (!currScope_) {
        throw std::runtime_error("CallStatNode: no current scope");
    }
    
    // First, evaluate arguments to get their types and values
    std::vector<VarInfo> argInfos;
    argInfos.reserve(node->call->args.size());
    
    // Evaluate all arguments and collect their VarInfo
    for (const auto& argExpr : node->call->args) {
        if (!argExpr) {
            throw std::runtime_error("CallStatNode: null argument expression");
        }
        argExpr->accept(*this);
        if (v_stack_.empty()) {
            throw std::runtime_error("CallStatNode: argument evaluation did not produce a value");
        }
        VarInfo argInfo = popValue();
        argInfos.push_back(argInfo);
    }
    
    // Build type-only VarInfo list for procedure resolution
    std::vector<VarInfo> typeOnlyArgs;
    typeOnlyArgs.reserve(argInfos.size());
    for (const auto& argInfo : argInfos) {
        typeOnlyArgs.push_back(VarInfo{"", argInfo.type, true});
    }
    
    // Resolve procedure using types only (semantic analysis should have validated this exists)
    ProcInfo* procInfo = nullptr;
    try {
        procInfo = currScope_->resolveProc(node->call->funcName, typeOnlyArgs, node->line);
    } catch (const CompileTimeException& e) {
        throw std::runtime_error("CallStatNode: procedure '" + node->call->funcName + "' not found or type mismatch: " + e.what());
    }
    if (!procInfo) {
        throw std::runtime_error("CallStatNode: procedure '" + node->call->funcName + "' not found");
    }
    
    // Look up the procedure function declaration (should exist from procedure definition)
    mlir::func::FuncOp procFunc = module_.lookupSymbol<mlir::func::FuncOp>(node->call->funcName);
    if (!procFunc) {
        throw std::runtime_error("CallStatNode: procedure function '" + node->call->funcName + "' not found in module");
    }
    
    // Build argument values for MLIR call
    std::vector<mlir::Value> callArgs;
    callArgs.reserve(argInfos.size());
    
    for (size_t i = 0; i < argInfos.size() && i < procInfo->params.size(); ++i) {
        const auto& param = procInfo->params[i];
        const auto& argInfo = argInfos[i];

        if (param.type.baseType == BaseType::TUPLE) {
            if (!argInfo.value) {
                throw std::runtime_error(
                    "CallStatNode: tuple argument has no value");
            }
            if (!param.isConst) {
                // var tuple: argument is pointer to struct, pass directly
                callArgs.push_back(argInfo.value);
            } else {
                // const tuple: pass struct by value
                mlir::Type structTy = getLLVMType(param.type);
                mlir::Value structVal = builder_.create<mlir::LLVM::LoadOp>(
                    loc_, structTy, argInfo.value);
                callArgs.push_back(structVal);
            }
        } else {
            if (!param.isConst) {
                // var parameter: pass memref directly
                if (!argInfo.value) {
                    throw std::runtime_error("CallStatNode: var parameter requires mutable argument (variable), but argument has no value");
                }
                mlir::Type argType = argInfo.value.getType();
                if (!argType.isa<mlir::MemRefType>()) {
                    throw std::runtime_error("CallStatNode: var parameter requires mutable argument (variable) with memref type");
                }
                callArgs.push_back(argInfo.value);
            } else {
                // const parameter: load value if it's a memref
                if (!argInfo.value) {
                    throw std::runtime_error("CallStatNode: argument has no value");
                }
                // Normalize to SSA (getSSAValue will load memref if needed)
                mlir::Value argVal = getSSAValue(argInfo);
                callArgs.push_back(argVal);
            }
        }
    }
    
    // Verify function signature matches what we're passing
    auto funcType = procFunc.getFunctionType();
    if (funcType.getNumInputs() != callArgs.size()) {
        throw std::runtime_error("CallStatNode: argument count mismatch for procedure '" + node->call->funcName + "'");
    }
    
    // Verify each argument type matches the function signature
    for (size_t i = 0; i < callArgs.size(); ++i) {
        mlir::Type expectedType = funcType.getInput(i);
        mlir::Type actualType = callArgs[i].getType();
        if (expectedType != actualType) {
            throw std::runtime_error("CallStatNode: type mismatch for argument " + std::to_string(i) + 
                " in procedure '" + node->call->funcName + "'");
        }
    }
    
    // Generate the call operation
    if (!builder_.getBlock()) {
        throw std::runtime_error("CallStatNode: builder has no current block");
    }
    
    builder_.create<mlir::func::CallOp>(loc_, procFunc, callArgs);
    
    // If procedure returns a value, we discard it (call statement doesn't use return value)
    // Per spec: "The return value from a procedure call can only be manipulated with unary operators"
    // Since this is a call statement, we just discard the result
}