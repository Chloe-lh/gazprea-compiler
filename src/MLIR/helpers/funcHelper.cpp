#include "MLIRgen.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"


mlir::func::FuncOp MLIRGen::beginFunctionDefinitionWithConstants(
    const ASTNode* funcOrProc,
    const std::string &name,
    const std::vector<VarInfo> &params,
    const CompleteType &returnType,
    Scope* &savedScope
    )
{
    auto func = beginFunctionDefinition(funcOrProc, name, params, returnType, savedScope);

    // Bind parameters: use constant values if available
    bindFunctionParametersWithConstants(func, params, funcOrProc->line);

    return func;
}

mlir::func::FuncOp MLIRGen::beginFunctionDefinition(
    const ASTNode* funcOrProc,
    const std::string &name,
    const std::vector<VarInfo> &params,
    const CompleteType &returnType,
    Scope*& savedScope
) {
    // Create or get declaration
    auto func = createFunctionDeclaration(name, params, returnType);

    // If function has no blocks, add entry block now
    mlir::Block* entry = nullptr;
    if (func.getBlocks().empty()) {
        entry = func.addEntryBlock();
    } else {
        entry = &func.front();
    }

    // Set allocas to the top of the entry block
    allocaBuilder_ = mlir::OpBuilder(entry, entry->begin());
    builder_.setInsertionPointToStart(entry);

    // Switch semantic scope using the semantic map if available
    savedScope = currScope_;
    auto it = scopeMap_->find(funcOrProc);
    if (it != scopeMap_->end()) {
        currScope_ = it->second;
    } else {
        throw std::runtime_error("MLIRGen: No semantic scope registered for '" + name + "'.");
    }

    return func;
}

mlir::func::FuncOp MLIRGen::createFunctionDeclaration(const std::string &name,
                                                     const std::vector<VarInfo> &params,
                                                     const CompleteType &returnType) {
    // Check if declaration already exists
    if (auto existing = module_.lookupSymbol<mlir::func::FuncOp>(name)) {
        return existing;
    }
    
    // Build function type
    std::vector<mlir::Type> argTys;
    argTys.reserve(params.size());
    for (const auto &p : params) {
        // For scalar var parameters -> use memref
        // For scalar const parameter -> scalar type (call by val)
        // For tuple parameters use llvm.ptr to struct
        if (p.type.baseType == BaseType::TUPLE) {
            mlir::Type structTy = getLLVMType(p.type);
            if (!p.isConst) {
                // var tuple: pointer to struct
                argTys.push_back(mlir::LLVM::LLVMPointerType::get(&context_));
            } else {
                // const tuple: struct by value
                argTys.push_back(structTy);
            }
        } else {
            if (!p.isConst) {
                mlir::Type elemTy = getLLVMType(p.type);
                argTys.push_back(mlir::MemRefType::get({}, elemTy));
            } else {
                argTys.push_back(getLLVMType(p.type));
            }
        }
    }

    std::vector<mlir::Type> resTys;
    if (returnType.baseType != BaseType::UNKNOWN) resTys.push_back(getLLVMType(returnType));

    auto ftype = builder_.getFunctionType(argTys, resTys);

    // Create function at module level
    auto* moduleBuilder = backend_.getBuilder().get();
    auto savedIP = moduleBuilder->saveInsertionPoint();
    moduleBuilder->setInsertionPointToStart(module_.getBody());
    auto func = moduleBuilder->create<mlir::func::FuncOp>(loc_, name, ftype);
    moduleBuilder->restoreInsertionPoint(savedIP);
    
    return func;
}

void MLIRGen::bindFunctionParametersWithConstants(mlir::func::FuncOp func, const std::vector<VarInfo> &params, int line) {
    if (func.getBlocks().empty()) return;
    mlir::Block &entry = func.front();

    for (size_t i = 0; i < params.size(); ++i) {
        const auto &p = params[i];
        VarInfo* vi = currScope_ ? currScope_->resolveVar(p.identifier,line) : nullptr;
        if (!vi) throw std::runtime_error("Codegen: missing parameter '" + p.identifier + "' in scope");
        
        mlir::Value argValue = entry.getArgument(i);

        // Tuple parameters use the LLVM struct representation and are not
        // lowered through memref.
        if (p.type.baseType == BaseType::TUPLE) {
            if (!p.isConst) {
                // var tuple: argument is already a pointer to the struct
                vi->value = argValue;
            } else {
                // const tuple: argument is the struct value; allocate storage
                // (if needed) and store into it.
                if (!vi->value) allocaVar(vi, line);
                builder_.create<mlir::LLVM::StoreOp>(loc_, argValue, vi->value);
            }
        } else {
            // Scalar parameters: var uses memref (passed by reference),
            // const is passed by value and stored into an alloca.
            if (!p.isConst && argValue.getType().isa<mlir::MemRefType>()) {
                vi->value = argValue;
            } else {
                if (!vi->value) allocaVar(vi, line);
                builder_.create<mlir::memref::StoreOp>(
                    loc_, argValue, vi->value, mlir::ValueRange{});
            }
        }
    }
}

void MLIRGen::lowerFunctionOrProcedureBody(const std::vector<VarInfo> &params,
                                           std::shared_ptr<BlockNode> body,
                                           const CompleteType &returnType,
                                           Scope* savedScope)
{
    // Save insertion point before visiting body (we're in the function's entry block)
    auto savedIP = builder_.saveInsertionPoint();
    
    // Get the function from the current block before visiting body
    auto* region = builder_.getBlock()->getParent();
    mlir::func::FuncOp funcOp = nullptr;
    if (region) {
        mlir::Operation* parentOp = region->getParentOp();
        while (parentOp) {
            if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(parentOp)) {
                funcOp = func;
                break;
            }
            parentOp = parentOp->getParentOp();
        }
    }
    
    if (body) body->accept(*this);
    
    // Restore insertion point to where we were (function's entry block)
    builder_.restoreInsertionPoint(savedIP);
    
    // Use the function we found before visiting the body
    if (!funcOp) {
        // Fallback: try to get it from current block
        auto* currentRegion = builder_.getBlock()->getParent();
        if (currentRegion) {
            mlir::Operation* parentOp = currentRegion->getParentOp();
            while (parentOp) {
                if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(parentOp)) {
                    funcOp = func;
                    break;
                }
                parentOp = parentOp->getParentOp();
            }
        }
    }
    
    if (!funcOp) {
        currScope_ = savedScope;
        return;
    }
    
    auto &entry = funcOp.front();

    // First pass: find at least one func.return and remember its block so we can later patch unterminated blocks.
    bool hasReturn = false;
    mlir::Block *firstReturnBlock = nullptr;
    for (auto &block : funcOp.getBody().getBlocks()) {
        if (!block.empty() &&
            block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            if (llvm::isa<mlir::func::ReturnOp>(block.back())) {
                hasReturn = true;
                if (!firstReturnBlock)
                    firstReturnBlock = &block;
            }
        }
    }

    // If there was no return, add a default one at the end of the entry
    // block for void functions
    builder_.setInsertionPointToEnd(&entry);
    if (!hasReturn) {
        if (returnType.baseType == BaseType::UNKNOWN) {
            builder_.create<mlir::func::ReturnOp>(loc_);
            hasReturn = true;
            firstReturnBlock = &entry;
        } else {
            throw std::runtime_error("missing return in non-void function");
        }
    }

    // Ensure every block in the function ends with a terminator. For any empty or unterminated block, branch to block w/ function return
    mlir::Block *exitBlock = firstReturnBlock ? firstReturnBlock : &entry;
    for (auto &block : funcOp.getBody().getBlocks()) {
        if (block.empty() ||
            !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            builder_.setInsertionPointToEnd(&block);
            builder_.create<mlir::cf::BranchOp>(loc_, exitBlock);
        }
    }

    currScope_ = savedScope;
}