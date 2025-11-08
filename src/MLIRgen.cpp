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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "CompileTimeExceptions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Casting.h"


#include <stdexcept>

MLIRGen::MLIRGen(BackEnd& backend): backend_(backend), builder_(*backend.getBuilder()), allocaBuilder_(*backend.getBuilder()), module_(backend.getModule()), context_(backend.getContext()), loc_(backend.getLoc())  {
    root_ = nullptr;
    currScope_ = nullptr;
    scopeMap_ = nullptr;
}

MLIRGen::MLIRGen(BackEnd& backend, Scope* rootScope, const std::unordered_map<const ASTNode*, Scope*>* scopeMap)
    : backend_(backend), builder_(*backend.getBuilder()), allocaBuilder_(*backend.getBuilder()), module_(backend.getModule()), context_(backend.getContext()), loc_(backend.getLoc()), root_(rootScope), currScope_(nullptr), scopeMap_(scopeMap) {
    // Ensure printf and global strings are created upfront
    if (!module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
        auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
        auto i32Ty = builder_.getI32Type();
        auto printfType = mlir::LLVM::LLVMFunctionType::get(i32Ty, ptrTy, true);
        builder_.create<mlir::LLVM::LLVMFuncOp>(loc_, "printf", printfType);
    }
    auto createGlobalStringIfMissing = [&](const char *str, const char *name) {
        if (!module_.lookupSymbol<mlir::LLVM::GlobalOp>(name)) {
            mlir::Type charType = builder_.getI8Type();
            auto strRef = mlir::StringRef(str, strlen(str) + 1);
            auto strType = mlir::LLVM::LLVMArrayType::get(charType, strRef.size());
            builder_.create<mlir::LLVM::GlobalOp>(loc_, strType, true,
                                    mlir::LLVM::Linkage::Internal, name,
                                    builder_.getStringAttr(strRef), 0);
        }
    };
    createGlobalStringIfMissing("%d\0", "intFormat");
    createGlobalStringIfMissing("%c\0", "charFormat");
    createGlobalStringIfMissing("%.2f\0", "floatFormat");
    createGlobalStringIfMissing("%s\0", "strFormat");
    createGlobalStringIfMissing("\n\0", "newline");
}


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

mlir::func::FuncOp MLIRGen::beginFunctionDefinition(const std::string &name,
                                                    const std::vector<VarInfo> &params,
                                                    const CompleteType &returnType,
                                                    Scope*& savedScope) {
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

    // Switch semantic scope: find child scope matching parameter names
    savedScope = currScope_;
    Scope* funcScope = nullptr;
    if (currScope_ && !currScope_->children().empty()) {
        if (!params.empty()) {
            for (const auto &ch : currScope_->children()) {
                const auto &syms = ch->symbols();
                bool ok = true;
                for (const auto &p : params) {
                    if (syms.find(p.identifier) == syms.end()) { ok = false; break; }
                }
                if (ok) { funcScope = ch.get(); break; }
            }
        }
        if (!funcScope && !currScope_->children().empty()) funcScope = currScope_->children().front().get();
    }
    if (funcScope) currScope_ = funcScope;

    return func;
}
mlir::func::FuncOp MLIRGen::beginFunctionDefinitionWithConstants(
    const std::string &name,
    const std::vector<VarInfo> &params,
    const CompleteType &returnType,
    Scope* &savedScope)
{
    auto func = beginFunctionDefinition(name, params, returnType, savedScope);

    // Bind parameters: use constant values if available
    bindFunctionParametersWithConstants(func, params);

    return func;
}

mlir::func::FuncOp MLIRGen::createFunctionDeclaration(const std::string &name,
                                                     const std::vector<VarInfo> &params,
                                                     const CompleteType &returnType) {
    // Build function type
    std::vector<mlir::Type> argTys;
    argTys.reserve(params.size());
    for (const auto &p : params) argTys.push_back(getLLVMType(p.type));

    std::vector<mlir::Type> resTys;
    if (returnType.baseType != BaseType::UNKNOWN) resTys.push_back(getLLVMType(returnType));

    auto ftype = builder_.getFunctionType(argTys, resTys);

    // If a declaration doesn't already exist, create one
    if (auto existing = module_.lookupSymbol<mlir::func::FuncOp>(name)) {
        return existing;
    }
    return builder_.create<mlir::func::FuncOp>(loc_, name, ftype);
}

void MLIRGen::bindFunctionParametersWithConstants(mlir::func::FuncOp func, const std::vector<VarInfo> &params) {
    if (func.getBlocks().empty()) return;
    mlir::Block &entry = func.front();

    for (size_t i = 0; i < params.size(); ++i) {
        const auto &p = params[i];
        VarInfo* vi = currScope_ ? currScope_->resolveVar(p.identifier) : nullptr;
        if (!vi) throw std::runtime_error("Codegen: missing parameter '" + p.identifier + "' in scope");
        // Ensure parameter has an allocated memref to store the incoming
        // entry-block argument. Constant folding stores are represented on
        // the AST (ExprNode::constant), not on VarInfo, so do not attempt
        // to read a 'constant' field here.
        if (!vi->value) allocaVar(vi);
        builder_.create<mlir::memref::StoreOp>(loc_, entry.getArgument(i), vi->value, mlir::ValueRange{});
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
    
    // Check if ANY block in the function body has a return terminator
    // func.return can only be in function blocks, not region blocks
    bool hasReturn = false;
    for (auto &block : funcOp.getBody().getBlocks()) {
        if (!block.empty() && block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            if (llvm::isa<mlir::func::ReturnOp>(block.back())) {
                hasReturn = true;
                break;
            }
        }
    }
    
    // Set insertion point to the end of the entry block
    builder_.setInsertionPointToEnd(&entry);
    
    // If no return was found, add one
    if (!hasReturn) {
        if (returnType.baseType == BaseType::UNKNOWN)
            builder_.create<mlir::func::ReturnOp>(loc_);
        else
            throw std::runtime_error("missing return in non-void function");
    }

    currScope_ = savedScope;
}

void MLIRGen::visit(FileNode* node) {
    // Initialize to semantic global scope (first child of root)
    currScope_ = root_;
    if (currScope_ && !currScope_->children().empty()) {
        currScope_ = currScope_->children().front().get();
    }

    // First pass: emit real const globals with constant initializers
    auto* moduleBuilder = backend_.getBuilder().get();
    auto savedIP = moduleBuilder->saveInsertionPoint();
    moduleBuilder->setInsertionPointToStart(module_.getBody());

    for (auto& n : node->stats) {
        auto tdecl = std::dynamic_pointer_cast<TypedDecNode>(n);
        if (!tdecl) continue;
        // Globals must be const and have an initializer
        if (tdecl->qualifier != "const" || !tdecl->init) {
            continue; // semantics should enforce, skip silently here
        }
        // Build constant attribute
        CompleteType gtype = tdecl->type_alias ? tdecl->type_alias->type : CompleteType(BaseType::UNKNOWN);
        mlir::Attribute initAttr = extractConstantValue(tdecl->init, gtype);
        if (!initAttr) {
            throw std::runtime_error("Global initializer must be a compile-time constant for '" + tdecl->name + "'.");
        }
        // Create the LLVM global
        (void) createGlobalVariable(tdecl->name, gtype, /*isConst=*/true, initAttr);
    }

    moduleBuilder->restoreInsertionPoint(savedIP);

    // Second pass: lower procedures/functions
    for (auto& n : node->stats) {
        if (std::dynamic_pointer_cast<ProcedureNode>(n) ||
            std::dynamic_pointer_cast<FuncStatNode>(n) ||
            std::dynamic_pointer_cast<FuncBlockNode>(n) ||
            std::dynamic_pointer_cast<FuncPrototypeNode>(n)) {
            n->accept(*this);
        }
    }
}

// functions
void MLIRGen::visit(FuncStatNode* node) {
    Scope* savedScope = nullptr;
    beginFunctionDefinitionWithConstants(node->name, node->parameters, node->returnType, savedScope);
    lowerFunctionOrProcedureBody(node->parameters, node->body, node->returnType, savedScope);
}

void MLIRGen::visit(FuncBlockNode* node) {
    Scope* savedScope = nullptr;
    beginFunctionDefinitionWithConstants(node->name, node->parameters, node->returnType, savedScope);
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

void MLIRGen::visit(FuncCallExpr* node) {
    // TODO: handle function calls
}

void MLIRGen::visit(ProcedureNode* node) {
    Scope* savedScope = nullptr;
    beginFunctionDefinitionWithConstants(node->name, node->params, node->returnType, savedScope);
    lowerFunctionOrProcedureBody(node->params, node->body, node->returnType, savedScope);
}

// Declarations / Globals helpers
mlir::Type MLIRGen::getLLVMType(const CompleteType& type) {
    // TODO: support tuples
    switch (type.baseType) {
        case BaseType::BOOL: return builder_.getI1Type();
        case BaseType::CHARACTER: return builder_.getI8Type();
        case BaseType::INTEGER: return builder_.getI32Type();
        case BaseType::REAL: return builder_.getF32Type();
        default:
            throw std::runtime_error("Unsupported global type: " + toString(type));
    }
}

mlir::Attribute MLIRGen::extractConstantValue(std::shared_ptr<ExprNode> expr, const CompleteType& targetType) {
    if (!expr) return nullptr;
    if (auto tn = std::dynamic_pointer_cast<TrueNode>(expr)) {
        return builder_.getIntegerAttr(builder_.getI1Type(), 1);
    }
    if (auto fn = std::dynamic_pointer_cast<FalseNode>(expr)) {
        return builder_.getIntegerAttr(builder_.getI1Type(), 0);
    }
    if (auto cn = std::dynamic_pointer_cast<CharNode>(expr)) {
        return builder_.getIntegerAttr(builder_.getI8Type(), static_cast<int>(cn->value));
    }
    if (auto in = std::dynamic_pointer_cast<IntNode>(expr)) {
        return builder_.getIntegerAttr(builder_.getI32Type(), in->value);
    }
    if (auto rn = std::dynamic_pointer_cast<RealNode>(expr)) {
        return builder_.getFloatAttr(builder_.getF32Type(), rn->value);
    }
    return nullptr;
}

mlir::Value MLIRGen::createGlobalVariable(const std::string& name, const CompleteType& type, bool isConst, mlir::Attribute initValue) {
    mlir::Type elemTy = getLLVMType(type);
    auto* moduleBuilder = backend_.getBuilder().get();
    auto savedIP = moduleBuilder->saveInsertionPoint();
    moduleBuilder->setInsertionPointToStart(module_.getBody());
    moduleBuilder->create<mlir::LLVM::GlobalOp>(
        loc_, elemTy, isConst, mlir::LLVM::Linkage::Internal, name, initValue, 0);
    moduleBuilder->restoreInsertionPoint(savedIP);
    return nullptr;
}


void MLIRGen::visit(TypedDecNode* node) {
    // Resolve variable declared by semantic analysis
    VarInfo* declaredVar = currScope_->resolveVar(node->name);

    // defensive sync for qualifier flag
    if (node->qualifier == "const") {
        declaredVar->isConst = true;
    } else if (node->qualifier == "var") {
        declaredVar->isConst = false;
    }

    // Ensure storage exists regardless of initializer
    if (!declaredVar->value) {
        allocaVar(declaredVar);
    }

    // Handle optional initializer + promotion
    if (node->init) {
        node->init->accept(*this);
        VarInfo literal = popValue();
        assignTo(&literal, declaredVar);
    }
}

/* Functionally the same as TypedDecNode except initializer is required */
void MLIRGen::visit(InferredDecNode* node) {
    if (!node->init) {
        throw std::runtime_error("FATAL: Inferred declaration without initializer.");
    }
    node->init->accept(*this); // Resolve init value

    VarInfo literal = popValue();
    VarInfo* declaredVar = currScope_->resolveVar(node->name);


    // Semantic analysis should have handled this - this is just in casse
    if (node->qualifier == "const") {
        declaredVar->isConst = true;
    } else if (node->qualifier == "var") {
        declaredVar->isConst = false;
    } else {
        throw StatementError(1, "Cannot infer variable '" + node->name + "' without qualifier."); // TODO: line number
    }
    
    assignTo(&literal, declaredVar);
}

void MLIRGen::visit(TupleTypedDecNode* node) {
    // Resolve variable declared by semantic analysis
    VarInfo* declaredVar = currScope_->resolveVar(node->name);

    // Ensure storage for tuple elements exists
    if (declaredVar->mlirSubtypes.empty()) {
        allocaVar(declaredVar);
    }

    // Handle optional initializer
    if (node->init) {
        node->init->accept(*this);
        VarInfo literal = popValue();
        assignTo(&literal, declaredVar);
    }
}

/* Resolve aliases using currScope_->resolveAlias */
void MLIRGen::visit(TypeAliasDecNode* node) {
    /* Nothing to do - already declared during semantic analysis. */
}

/* Resolve aliases using currScope_->resolveAlias */
void MLIRGen::visit(TypeAliasNode* node) {
    /* Nothing to do - already declared during semantic analysis. */
}

/* Resolve aliases using currScope_->resolveAlias */
void MLIRGen::visit(TupleTypeAliasNode* node) {
    /* Nothing to do - already declared during semantic analysis. */
}

// Statements
void MLIRGen::visit(AssignStatNode* node) {
    if (!node->expr) throw std::runtime_error("FATAL: No expr for assign stat found"); 
    node->expr->accept(*this);
    VarInfo* to = currScope_->resolveVar(node->name);
    VarInfo from = popValue();

    if (to->isConst) {
        throw AssignError(1, "Cannot assign to const variable '" + to->identifier + "'.");
    }

    assignTo(&from, to);
}

void MLIRGen::visit(OutputStatNode* node) {
    
    if (!node->expr) {
        return;
    }
    
    // Handle string literals
    if (auto strNode = std::dynamic_pointer_cast<StringNode>(node->expr)) {
        auto printfFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
        auto formatString = module_.lookupSymbol<mlir::LLVM::GlobalOp>("strFormat");
        if (!printfFunc || !formatString) {
            throw std::runtime_error("MLIRGen::OutputStat: missing printf or strFormat.");
        }
        // Create/find a global for the string literal in curr scope
        std::string symName = std::string("strlit_") + std::to_string(std::hash<std::string>{}(strNode->value));
        auto existing = module_.lookupSymbol<mlir::LLVM::GlobalOp>(symName);
        if (!existing) {
            auto* moduleBuilder = backend_.getBuilder().get();
            auto savedIP = moduleBuilder->saveInsertionPoint();
            moduleBuilder->setInsertionPointToStart(module_.getBody());
            mlir::Type charTy = builder_.getI8Type();
            mlir::StringRef sref(strNode->value.c_str(), strNode->value.size() + 1);
            auto arrTy = mlir::LLVM::LLVMArrayType::get(charTy, sref.size());
            moduleBuilder->create<mlir::LLVM::GlobalOp>(loc_, arrTy, /*constant=*/true,
                mlir::LLVM::Linkage::Internal, symName, builder_.getStringAttr(sref), 0);
            moduleBuilder->restoreInsertionPoint(savedIP);
            existing = module_.lookupSymbol<mlir::LLVM::GlobalOp>(symName);
        }
        auto fmtPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, formatString);
        auto strPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, existing);
        builder_.create<mlir::LLVM::CallOp>(loc_, printfFunc, mlir::ValueRange{fmtPtr, strPtr});
        return;
    }

    // Evaluate the expression to get the value to print
    node->expr->accept(*this);
    VarInfo exprVarInfo = popValue();

    // Load the value from its memref if needed. Some visitors may push
    // scalar mlir::Value directly (non-memref), so accept both forms.
    mlir::Value loadedValue;
    if (exprVarInfo.value.getType().isa<mlir::MemRefType>()) {
        loadedValue = builder_.create<mlir::memref::LoadOp>(
        loc_, exprVarInfo.value, mlir::ValueRange{});
    } else {
        loadedValue = exprVarInfo.value;
    }

    // Determine format string name and get format string/printf upfront
    const char* formatStrName = nullptr;
    switch (exprVarInfo.type.baseType) {
        case BaseType::BOOL:
        case BaseType::INTEGER:
            formatStrName = "intFormat";
            break;
        case BaseType::REAL:
            formatStrName = "floatFormat";
            break;
        case BaseType::CHARACTER:
            formatStrName = "charFormat";
            break;
        case BaseType::STRING:
            formatStrName = "strFormat";
            break;
        default:
            throw std::runtime_error("MLIRGen::OutputStat: Unsupported type for printing.");
    }

    // Lookup the format string and printf function (these are module-level symbols)
    auto formatString = module_.lookupSymbol<mlir::LLVM::GlobalOp>(formatStrName);
    auto printfFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    
    if (!formatString || !printfFunc) {
        throw std::runtime_error("MLIRGen::OutputStat: Format string or printf function not found.");
    }
    
    // Get the address of the format string - create this before any value transformations
    mlir::Value formatStringPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, formatString);
    
    // Now transform the value if needed (extensions)
    mlir::Value valueToPrint = loadedValue;
    switch (exprVarInfo.type.baseType) {
        case BaseType::BOOL:
            // Extend boolean to i32 for printing
            valueToPrint = builder_.create<mlir::arith::ExtUIOp>(
                loc_, builder_.getI32Type(), loadedValue);
            break;
        case BaseType::REAL:
            // Extend float to f64 for printing
            valueToPrint = builder_.create<mlir::arith::ExtFOp>(
                loc_, builder_.getF64Type(), loadedValue);
            break;
        case BaseType::CHARACTER:
            // Extend character to i32 for printing
            valueToPrint = builder_.create<mlir::arith::ExtSIOp>(
                loc_, builder_.getI32Type(), loadedValue);
            break;
        case BaseType::INTEGER:
            // No extension needed for integer
            break;
        case BaseType::STRING:
            // For non-literal strings we'd expect a pointer; assume loadedValue is already a ptr
            break;
        default:
            break;
    }
    
    // Create the printf call with format string and value
    // Both operands (formatStringPtr and valueToPrint) must dominate this operation
    builder_.create<mlir::LLVM::CallOp>(
        loc_, 
        printfFunc, 
        mlir::ValueRange{formatStringPtr, valueToPrint}
    );
}

void MLIRGen::visit(InputStatNode* node) { throw std::runtime_error("InputStatNode not implemented"); }
void MLIRGen::visit(BreakStatNode* node) {
    if (loopContexts_.empty()) {
        throw std::runtime_error("BreakStatNode: break statement outside of loop");
    }
    // Set the break flag to false, which will cause the loop to exit
    auto& loopCtx = loopContexts_.back();
    mlir::Type i1Type = builder_.getI1Type();
    auto falseAttr = builder_.getIntegerAttr(i1Type, 0);
    auto falseConst = builder_.create<mlir::arith::ConstantOp>(loc_, i1Type, falseAttr);
    builder_.create<mlir::memref::StoreOp>(loc_, falseConst.getResult(), loopCtx.breakFlag, mlir::ValueRange{});
    // Note: We can't directly yield from the loop body here since we're inside an if region
    // The break flag will cause the loop to exit on the next condition check
    // This means statements after the if will still execute in the current iteration
    // This is a known limitation of using MLIR SCF for imperative break statements
}

void MLIRGen::visit(ContinueStatNode* node) {
    if (loopContexts_.empty()) {
        throw std::runtime_error("ContinueStatNode: continue statement outside of loop");
    }
    // Yield immediately to continue to the next iteration (skip rest of body)
    builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{});
}
void MLIRGen::visit(ReturnStatNode* node) {
    const CompleteType* retTy = currScope_ ? currScope_->getReturnType() : nullptr;
    if (!retTy || retTy->baseType == BaseType::UNKNOWN) {
        builder_.create<mlir::func::ReturnOp>(loc_);
        return;
    }
    if (!node->expr) {
        throw std::runtime_error("Codegen: missing return value for non-void procedure/function");
    }
    node->expr->accept(*this);
    VarInfo v = popValue();
    VarInfo promoted = promoteType(&v, const_cast<CompleteType*>(retTy));
    mlir::Value loaded = builder_.create<mlir::memref::LoadOp>(loc_, promoted.value, mlir::ValueRange{});
    builder_.create<mlir::func::ReturnOp>(loc_, loaded);
}
void MLIRGen::visit(CallStatNode* node) { throw std::runtime_error("CallStatNode not implemented"); }
void MLIRGen::visit(IfNode* node) {
    
    // Evaluate the condition expression in the current block
    node->cond->accept(*this);
    VarInfo condVarInfo = popValue();
    
    // Load the condition value from its memref in the current block
    mlir::Value conditionValue = builder_.create<mlir::memref::LoadOp>(
        loc_, condVarInfo.value, mlir::ValueRange{});

    // Determine if we have an else branch
    bool hasElse = (node->elseBlock != nullptr) || (node->elseStat != nullptr);
    
    // Create the scf.if operation
    auto ifOp = builder_.create<mlir::scf::IfOp>(loc_, conditionValue, hasElse);

    // Build the 'then' region
    {
        auto& thenBlock = ifOp.getThenRegion().front();
        
        // Set insertion point to the start of the then block
        builder_.setInsertionPointToStart(&thenBlock);
        
        // Update allocaBuilder to point to the then block so allocas are created here
        auto savedAllocaBuilder = allocaBuilder_;
        allocaBuilder_ = mlir::OpBuilder(&thenBlock, thenBlock.begin());
        
        // Visit the then branch
    if (node->thenBlock) {
        node->thenBlock->accept(*this);
    } else if (node->thenStat) {
        node->thenStat->accept(*this);
        }
        
        allocaBuilder_ = savedAllocaBuilder;
        
        // Insert yield only if there isn't already a terminator (e.g., from break/continue)
        if (thenBlock.empty() || !thenBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            if (!thenBlock.empty()) {
                builder_.setInsertionPointAfter(&thenBlock.back());
            } else {
                builder_.setInsertionPointToStart(&thenBlock);
    }
    builder_.create<mlir::scf::YieldOp>(loc_);
        }
    }

    // Build the 'else' region if it exists
    if (hasElse) {
        auto& elseBlock = ifOp.getElseRegion().front();
        
        // Set insertion point to the start of the else block
        builder_.setInsertionPointToStart(&elseBlock);
        
        // Update allocaBuilder to point to the else block so allocas are created here
        auto savedAllocaBuilder = allocaBuilder_;
        allocaBuilder_ = mlir::OpBuilder(&elseBlock, elseBlock.begin());
        
        // Visit the else branch
        if (node->elseBlock) {
            node->elseBlock->accept(*this);
        } else if (node->elseStat) {
            node->elseStat->accept(*this);
        }

        allocaBuilder_ = savedAllocaBuilder;
        
        // Insert yield only if there isn't already a terminator (e.g., from break/continue)
        if (elseBlock.empty() || !elseBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            if (!elseBlock.empty()) {
                builder_.setInsertionPointAfter(&elseBlock.back());
            } else {
                builder_.setInsertionPointToStart(&elseBlock);
        }
        builder_.create<mlir::scf::YieldOp>(loc_);
        }
    }

    // Restore insertion point after the if operation
    builder_.setInsertionPointAfter(ifOp);
}
void MLIRGen::visit(LoopNode* node) {
    if (!node->body) {
        throw std::runtime_error("LoopNode: loop body is null");
    }

    // Create a break flag (true = continue, false = break)
    CompleteType boolType = CompleteType(BaseType::BOOL);
    VarInfo breakFlagVar = VarInfo(boolType);
    allocaVar(&breakFlagVar);
    mlir::Type i1Type = builder_.getI1Type();
    auto trueAttr = builder_.getIntegerAttr(i1Type, 1);
    auto trueConst = builder_.create<mlir::arith::ConstantOp>(loc_, i1Type, trueAttr);
    builder_.create<mlir::memref::StoreOp>(loc_, trueConst.getResult(), breakFlagVar.value, mlir::ValueRange{});
    
    // Create scf.while operation (empty initially)
    auto whileOp = builder_.create<mlir::scf::WhileOp>(loc_, mlir::TypeRange{}, mlir::ValueRange{});
    
    // Save insertion point AFTER creating the loop 
    auto savedIPAfterLoop = builder_.saveInsertionPoint();
    
    // Build the "before" region (condition check)
    mlir::Region& beforeRegion = whileOp.getBefore();
    mlir::Block* beforeBlock = new mlir::Block();
    beforeRegion.push_back(beforeBlock);
    
    {
        auto savedIP = builder_.saveInsertionPoint();
        builder_.setInsertionPointToStart(beforeBlock);
        
        mlir::Value condValue;
        mlir::Value breakFlagVal = builder_.create<mlir::memref::LoadOp>(loc_, breakFlagVar.value, mlir::ValueRange{});
        
        if (node->kind == LoopKind::While && node->cond) {
            // Re-evaluate condition in the before region
            node->cond->accept(*this);
            VarInfo condVarInfo = popValue();
            mlir::Value condVal = builder_.create<mlir::memref::LoadOp>(loc_, condVarInfo.value, mlir::ValueRange{});
            // Combine condition with break flag: loop continues if condition is true AND break flag is true
            condValue = builder_.create<mlir::arith::AndIOp>(loc_, condVal, breakFlagVal);
        } else {
            // Plain loop: continue if break flag is true
            condValue = breakFlagVal;
        }
        
        // Yield the condition value
        builder_.create<mlir::scf::ConditionOp>(loc_, condValue, mlir::ValueRange{});
        
        builder_.restoreInsertionPoint(savedIP);
    }
    
    // Build the "after" region (loop body)
    mlir::Region& afterRegion = whileOp.getAfter();
    mlir::Block* afterBlock = new mlir::Block();
    afterRegion.push_back(afterBlock);
    
    {
        auto savedIP = builder_.saveInsertionPoint();
        builder_.setInsertionPointToStart(afterBlock);
        
        // Reset break flag to true at the start of each iteration
        auto trueConstAfter = builder_.create<mlir::arith::ConstantOp>(loc_, i1Type, trueAttr);
        builder_.create<mlir::memref::StoreOp>(loc_, trueConstAfter.getResult(), breakFlagVar.value, mlir::ValueRange{});
        
        // Push loop context for break/continue
        LoopContext loopCtx;
        loopCtx.exitBlock = nullptr; // Not used with scf.while (break just yields)
        loopCtx.continueBlock = nullptr; // Not used with scf.while (continue just yields)
        loopCtx.breakFlag = breakFlagVar.value;
        loopContexts_.push_back(loopCtx);
        
        // Visit the loop body
        node->body->accept(*this);
        
        // Pop loop context
        loopContexts_.pop_back();
        
        // Only add yield if there's no existing terminator (break/continue might have added one)
        if (afterBlock->empty() || !afterBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            // Yield to continue the loop
            if (!afterBlock->empty()) {
                builder_.setInsertionPointAfter(&afterBlock->back());
        } else {
                builder_.setInsertionPointToStart(afterBlock);
            }
            builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{});
        }
        
        builder_.restoreInsertionPoint(savedIP);
    }
    
    // Restore insertion point to after the loop was created
    // This ensures the next statement (like return) is visited after the loop
    builder_.restoreInsertionPoint(savedIPAfterLoop);
}

void MLIRGen::visit(BlockNode* node) { 
    // Enter the corresponding semantic child scope if present
    Scope* saved = currScope_;
    if (scopeMap_) {
        auto it = scopeMap_->find(node);
        if (it != scopeMap_->end()) {
            currScope_ = it->second;
        }
    }

    for (const auto& d : node->decs) if (d) d->accept(*this);
    
    // Process statements, wrapping remaining ones in conditional if we're in a loop
    bool inLoop = !loopContexts_.empty();
    for (size_t i = 0; i < node->stats.size(); ++i) {
        if (node->stats[i]) {
            node->stats[i]->accept(*this);
            
            // If we're in a loop and there are more statements, 
            // wrap them in an scf.if that only executes if break flag is TRUE (continue)
            if (inLoop && i + 1 < node->stats.size()) {
                auto& loopCtx = loopContexts_.back();
                mlir::Value breakFlagVal = builder_.create<mlir::memref::LoadOp>(loc_, loopCtx.breakFlag, mlir::ValueRange{});
                
                // Create if: execute remaining statements only if flag is TRUE (not broken)
                auto breakCheckOp = builder_.create<mlir::scf::IfOp>(loc_, breakFlagVal, false);
                
                // The "then" branch: execute remaining statements
                auto& continueBlock = breakCheckOp.getThenRegion().front();
                builder_.setInsertionPointToStart(&continueBlock);
                
                // Process remaining statements inside this conditional
                for (size_t j = i + 1; j < node->stats.size(); ++j) {
                    if (node->stats[j]) {
                        node->stats[j]->accept(*this);
                    }
                }
                
                // Add yield only if there isn't already a terminator
                if (continueBlock.empty() || !continueBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
                    if (!continueBlock.empty()) {
                        builder_.setInsertionPointAfter(&continueBlock.back());
                    }
                    builder_.create<mlir::scf::YieldOp>(loc_);
                }
                
                // We've processed all remaining statements, so break out
                break;
            }
        }
    }

    currScope_ = saved;
}

// Expressions / Operators
// 
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

    if (!varInfo) {
        throw SymbolError(1, "Semantic Analysis: Variable '" + node->id + "' not defined.");
    }

    // If the variable doesn't have a value, it's likely a global variable
    // For globals, we need to create operations to access them
    if (!varInfo->value) {
        // Look up the global variable in the module
        auto globalOp = module_.lookupSymbol<mlir::LLVM::GlobalOp>(node->id);
        if (!globalOp) {
        throw SymbolError(1, "Semantic Analysis: Variable '" + node->id + "' not initialized.");
    }

        // Get the address of the global variable
        mlir::Value globalAddr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
        
        // Get the element type from the global (GlobalOp.getType() returns the element type)
        mlir::Type elementType = globalOp.getType();
        
        // Create a temporary alloca to hold the loaded value
        // This allows us to use the same memref-based code path
        VarInfo tempVarInfo = VarInfo(varInfo->type);
        allocaLiteral(&tempVarInfo);
        
        // Load from global using LLVM::LoadOp
        mlir::Value loadedValue = builder_.create<mlir::LLVM::LoadOp>(
            loc_, elementType, globalAddr);
        
        // Store the loaded value into the temporary memref
        builder_.create<mlir::memref::StoreOp>(loc_, loadedValue, tempVarInfo.value, mlir::ValueRange{});
        
        pushValue(tempVarInfo);
        return;
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

void MLIRGen::assignTo(VarInfo* literal, VarInfo* variable) {
    // Tuple assignment: element-wise store with implicit scalar promotions
    if (variable->type.baseType == BaseType::TUPLE) {
        if (literal->type.baseType != BaseType::TUPLE) {
            throw AssignError(1, "Cannot assign non-tuple to tuple variable '");
        }
        // Ensure destination tuple storage exists
        if (variable->mlirSubtypes.empty()) {
            allocaVar(variable);
        }
        // Ensure literal has alloca'd elements
        if (literal->mlirSubtypes.empty()) {
            throw std::runtime_error("FATAL: Assigning to '" + variable->identifier + "' with tuple that has no mlirSubtypes.");
        }
        if (literal->type.subTypes.size() != variable->type.subTypes.size()) {
            throw AssignError(1, "Tuple arity mismatch in assignment.");
        }
        for (size_t i = 0; i < variable->mlirSubtypes.size(); ++i) {
            VarInfo srcElem = literal->mlirSubtypes[i];
            VarInfo& dstElem = variable->mlirSubtypes[i];
            // Promote if needed (supports int->real, no-op otherwise)
            VarInfo promoted = promoteType(&srcElem, &dstElem.type);
            mlir::Value loaded = builder_.create<mlir::memref::LoadOp>(loc_, promoted.value, mlir::ValueRange{});
            builder_.create<mlir::memref::StoreOp>(loc_, loaded, dstElem.value, mlir::ValueRange{});
        }
        return;
    }

    // Scalar assignment
    // ensure var has a memref allocated
    if (!variable->value) {
        allocaVar(variable);
    }

    VarInfo promoted = promoteType(literal, &variable->type); // handle type promotions + errors

    mlir::Value loadedVal = builder_.create<mlir::memref::LoadOp>(
        loc_, promoted.value, mlir::ValueRange{}
    );
    builder_.create<mlir::memref::StoreOp>(
        loc_, loadedVal, variable->value, mlir::ValueRange{}
    );
}

void MLIRGen::allocaLiteral(VarInfo* varInfo) {
    varInfo->isConst = true;
    allocaVar(varInfo);
}

// Create a VarInfo that contains an allocated memref with the compile-time
// constant stored. Supports scalar types (int, real, bool, char). Throws on
// unsupported types.
VarInfo MLIRGen::createLiteralFromConstant(const ConstantValue &cv, const CompleteType &type) {
    VarInfo lit(type);
    // allocate a literal container (memref) and mark it const
    allocaLiteral(&lit);

    switch (cv.type.baseType) {
        case BaseType::INTEGER: {
            auto i32 = builder_.getI32Type();
            int64_t v = std::get<int64_t>(cv.value);
            auto c = builder_.create<mlir::arith::ConstantOp>(loc_, i32, builder_.getIntegerAttr(i32, static_cast<int64_t>(v)));
            builder_.create<mlir::memref::StoreOp>(loc_, c, lit.value, mlir::ValueRange{});
            break;
        }
        case BaseType::REAL: {
            auto f32 = builder_.getF32Type();
            double dv = std::get<double>(cv.value);
            auto c = builder_.create<mlir::arith::ConstantOp>(loc_, f32, builder_.getFloatAttr(f32, static_cast<float>(dv)));
            builder_.create<mlir::memref::StoreOp>(loc_, c, lit.value, mlir::ValueRange{});
            break;
        }
        case BaseType::BOOL: {
            auto i1 = builder_.getI1Type();
            bool bv = std::get<bool>(cv.value);
            auto c = builder_.create<mlir::arith::ConstantOp>(loc_, i1, builder_.getIntegerAttr(i1, bv ? 1 : 0));
            builder_.create<mlir::memref::StoreOp>(loc_, c, lit.value, mlir::ValueRange{});
            break;
        }
        case BaseType::CHARACTER: {
            auto i8 = builder_.getI8Type();
            char ch = std::get<char>(cv.value);
            auto c = builder_.create<mlir::arith::ConstantOp>(loc_, i8, builder_.getIntegerAttr(i8, static_cast<int>(ch)));
            builder_.create<mlir::memref::StoreOp>(loc_, c, lit.value, mlir::ValueRange{});
            break;
        }
        default:
            throw std::runtime_error("createLiteralFromConstant: unsupported constant type");
    }

    return lit;
}

bool MLIRGen::tryEmitConstantForNode(ExprNode* node) {
    if (!node) return false;
    if (!node->constant.has_value()) return false;
    try {
        VarInfo lit = createLiteralFromConstant(node->constant.value(), node->type);
        pushValue(lit);
        return true;
    } catch (...) {
        // unsupported constant type or codegen error; fall back to normal lowering
        return false;
    }
}

void MLIRGen::allocaVar(VarInfo* varInfo) {
    mlir::Block *entryBlock = allocaBuilder_.getBlock();
    if (!entryBlock) {
        entryBlock = builder_.getBlock();
    }
    if (!entryBlock) {
        throw std::runtime_error("allocaVar: no available entry block for allocation");
    }

    mlir::OpBuilder::InsertionGuard guard(allocaBuilder_);
    auto insertPos = entryBlock->begin();
    while (insertPos != entryBlock->end() && llvm::isa<mlir::memref::AllocaOp>(&*insertPos)) {
        ++insertPos;
    }
    allocaBuilder_.setInsertionPoint(entryBlock, insertPos);

    switch (varInfo->type.baseType) {
        case BaseType::BOOL:
            varInfo->value = allocaBuilder_.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI1Type()));
            break;
        case BaseType::CHARACTER:
            varInfo->value = allocaBuilder_.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI8Type()));
            break;
        case BaseType::INTEGER:
            varInfo->value = allocaBuilder_.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI32Type()));
            break;
        case BaseType::REAL:
            varInfo->value = allocaBuilder_.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getF32Type()));
            break;

        case (BaseType::TUPLE):
            if (varInfo->type.subTypes.size() < 2) {
                throw SizeError(1, "Error: Tuple must have at least 2 elements.");
            }
            for (CompleteType& subtype: varInfo->type.subTypes) {
                VarInfo mlirSubtype = VarInfo(subtype);
                mlirSubtype.isConst = varInfo->isConst; // Copy 'const'ness from parent

                // Copy over type info into VarInfo's subtypes
                varInfo->mlirSubtypes.emplace_back(
                    mlirSubtype
                );
                allocaVar(&varInfo->mlirSubtypes.back());
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

/* Only allows implicit promotion from integer -> real. throws AssignError otherwise. */
VarInfo MLIRGen::promoteType(VarInfo* from, CompleteType* toType) {
    // No-op when types are identical
    if (from->type == *toType) {
        return *from;
    }

    // Only support integer -> real promotion
    if (from->type.baseType == BaseType::INTEGER && toType->baseType == BaseType::REAL) {
        VarInfo to = VarInfo(*toType);
        allocaLiteral(&to);
        mlir::Value i32Val = builder_.create<mlir::memref::LoadOp>(loc_, from->value, mlir::ValueRange{});
        mlir::Value fVal = builder_.create<mlir::arith::SIToFPOp>(loc_, builder_.getF32Type(), i32Val);
        builder_.create<mlir::memref::StoreOp>(loc_, fVal, to.value, mlir::ValueRange{});
        return to;
    }

    throw AssignError(1, std::string("Codegen: unsupported promotion from '") +
        toString(from->type) + "' to '" + toString(*toType) + "'.");
}


void MLIRGen::visit(ParenExpr* node) {
    node->expr->accept(*this);
}

void MLIRGen::visit(UnaryExpr* node) {
    if (tryEmitConstantForNode(node)) return;
    node->operand->accept(*this);
    VarInfo operand = popValue();
    // Ensure we operate on a scalar: load from memref if needed
    mlir::Value operandVal;
    if (operand.value.getType().isa<mlir::MemRefType>()) {
        operandVal = builder_.create<mlir::memref::LoadOp>(loc_, operand.value, mlir::ValueRange{});
    } else {
        operandVal = operand.value;
    }

    if (node->op == "-") {
        auto zero = builder_.create<mlir::arith::ConstantOp>(
            loc_, operandVal.getType(), builder_.getZeroAttr(operandVal.getType()));
        auto result = builder_.create<mlir::arith::SubIOp>(loc_, zero, operandVal);

        // Store scalar result into a memref-backed VarInfo and push
        VarInfo outVar(operand.type);
        allocaLiteral(&outVar);
        builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
        outVar.identifier = "";
        pushValue(outVar);
        return;
    }

    // No-op: push original operand through
    operand.identifier = "";
    pushValue(operand);
}

void MLIRGen::visit(ExpExpr* node) {
    if (tryEmitConstantForNode(node)) return;
    node->left->accept(*this);
    VarInfo left = popValue();
    mlir::Value lhs = left.value;
    node->right->accept(*this);
    VarInfo right = popValue();
    mlir::Value rhs = right.value;

    bool isInt = lhs.getType().isa<mlir::IntegerType>();

    // Promote to float if needed
    if (isInt) {
        auto f32Type = builder_.getF32Type();
        lhs = builder_.create<mlir::arith::SIToFPOp>(loc_, f32Type, lhs);
        rhs = builder_.create<mlir::arith::SIToFPOp>(loc_, f32Type, rhs);
    }

    // Error check: 0 raised to negative power
    auto zero = builder_.create<mlir::arith::ConstantOp>(
            loc_, lhs.getType(), builder_.getZeroAttr(lhs.getType()));
    
    mlir::Value isZero = builder_.create<mlir::arith::CmpFOp>(
            loc_, mlir::arith::CmpFPredicate::OEQ, lhs, zero);
    
    mlir::Value ltZero = builder_.create<mlir::arith::CmpFOp>(
            loc_, mlir::arith::CmpFPredicate::OLT, rhs, zero);
    
    mlir::Value invalidExp = builder_.create<mlir::arith::AndIOp>(loc_, isZero, ltZero);

    auto parentBlock = builder_.getBlock();
    auto errorBlock = parentBlock->splitBlock(builder_.getInsertionPoint());
    auto continueBlock = errorBlock->splitBlock(errorBlock->begin());
    
    builder_.create<mlir::cf::CondBranchOp>(loc_, invalidExp, errorBlock, mlir::ValueRange{}, continueBlock, mlir::ValueRange{});

    builder_.setInsertionPointToStart(errorBlock);
    auto errorMsg = builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getStringAttr("Math Error: 0 cannot be raised to a negative power."));
    builder_.create<mlir::func::CallOp>(loc_, "MathError", mlir::TypeRange{}, mlir::ValueRange{errorMsg});
    
    builder_.setInsertionPointToStart(continueBlock);

    mlir::Value result = builder_.create<mlir::math::PowFOp>(loc_, lhs, rhs);

    // If original operands were int, apply math.floor and cast back to int
    if (isInt) {
        mlir::Value floored = builder_.create<mlir::math::FloorOp>(loc_, result);
        auto intType = builder_.getI32Type();
        result = builder_.create<mlir::arith::FPToSIOp>(loc_, intType, floored);
    }

    // Assume both operands are of same type. Create a memref-backed VarInfo
    // to hold the scalar result so later passes can rely on memref semantics.
    VarInfo outVar(left.type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}

void MLIRGen::visit(MultExpr* node){
    if (tryEmitConstantForNode(node)) return;

    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    mlir::Value left = leftInfo.value;
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    mlir::Value right = rightInfo.value;

    if(node->op == "/" || node->op == "%"){
        auto zero = builder_.create<mlir::arith::ConstantOp>(
            loc_, right.getType(), builder_.getZeroAttr(right.getType()));

        // Compare right == 0
        mlir::Value isZero;
        if (right.getType().isa<mlir::IntegerType>()) {
            isZero = builder_.create<mlir::arith::CmpIOp>(
                loc_, mlir::arith::CmpIPredicate::eq, right, zero);
        } else if (right.getType().isa<mlir::FloatType>()) {
            isZero = builder_.create<mlir::arith::CmpFOp>(
                loc_, mlir::arith::CmpFPredicate::OEQ, right, zero);
        }

        // Create block for error and block for normal division
        auto parentBlock = builder_.getBlock();
        auto errorBlock = parentBlock->splitBlock(builder_.getInsertionPoint());
        auto continueBlock = errorBlock->splitBlock(errorBlock->begin());

        // Conditional branch
        builder_.create<mlir::cf::CondBranchOp>(loc_, isZero, errorBlock, mlir::ValueRange{}, continueBlock, mlir::ValueRange{});

        // Error block: call runtime error function
        builder_.setInsertionPointToStart(errorBlock);
        auto errorMsg = builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getStringAttr("Divide by zero error"));
        builder_.create<mlir::func::CallOp>(loc_, "MathError", mlir::TypeRange{}, mlir::ValueRange{errorMsg});

        // Continue block: do the division
        builder_.setInsertionPointToStart(continueBlock);
    }

    mlir::Value result;
    if(left.getType().isa<mlir::IntegerType>()) {
        if (node->op == "*") {
            result = builder_.create<mlir::arith::MulIOp>(loc_, left, right);
        } else if (node->op == "/") {
            result = builder_.create<mlir::arith::DivSIOp>(loc_, left, right);
        } else if (node->op == "%") {
            result = builder_.create<mlir::arith::RemSIOp>(loc_, left, right);
        }
    } else if(left.getType().isa<mlir::FloatType>()) {
        if (node->op == "*") {
            result = builder_.create<mlir::arith::MulFOp>(loc_, left, right);
        } else if (node->op == "/") {
            result = builder_.create<mlir::arith::DivFOp>(loc_, left, right);
        }
    } else {
        throw std::runtime_error("MLIRGen Error: Unsupported type for multiplication.");
    }

    // Wrap scalar result into a memref-backed VarInfo
    VarInfo outVar(leftInfo.type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}

void MLIRGen::visit(AddExpr* node){
    // If the node has a compile-time constant, emit it directly and push.
    if (tryEmitConstantForNode(node)) return;

    // Evaluate left then right (visitors push values onto the stack)
    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    mlir::Value left = leftInfo.value;
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    mlir::Value right = rightInfo.value;

    // helper: load if this value is a memref
    auto loadIfMemref = [&](mlir::Value v) -> mlir::Value {
        if (v.getType().isa<mlir::MemRefType>()) {
            return builder_.create<mlir::memref::LoadOp>(loc_, v, mlir::ValueRange{});
        }
        return v;
    };

    mlir::Value leftLoaded  = loadIfMemref(left);
    mlir::Value rightLoaded = loadIfMemref(right);

    mlir::Value result;
    if (leftLoaded.getType().isa<mlir::IntegerType>()) {
        if (node->op == "+") {
            result = builder_.create<mlir::arith::AddIOp>(loc_, leftLoaded, rightLoaded);
        } else if (node->op == "-") {
            result = builder_.create<mlir::arith::SubIOp>(loc_, leftLoaded, rightLoaded);
        }
    } else if (leftLoaded.getType().isa<mlir::FloatType>()) {
        if (node->op == "+") {
            result = builder_.create<mlir::arith::AddFOp>(loc_, leftLoaded, rightLoaded);
        } else if (node->op == "-") {
            result = builder_.create<mlir::arith::SubFOp>(loc_, leftLoaded, rightLoaded);
        }
    } else {
        throw std::runtime_error("MLIRGen Error: Unsupported type for addition.");
    }

    // Wrap scalar result into a memref-backed VarInfo
    VarInfo outVar(leftInfo.type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}

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

// Helper functions for equality
mlir::Value mlirScalarEquals(mlir::Value left, mlir::Value right, mlir::Location loc, mlir::OpBuilder& builder) {
    mlir::Type type = left.getType();
    if (type.isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, left, right);
    } else if (type.isa<mlir::FloatType>()) {
        return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, left, right);
    } else {
        throw std::runtime_error("mlirScalarEquals: Unsupported type for equality");
    }
}

mlir::Value mlirTupleEquals(VarInfo& leftInfo, VarInfo& rightInfo, mlir::Location loc, mlir::OpBuilder& builder) {
    // Both leftInfo and rightInfo are VarInfo with mlirSubtypes
    int numElements = leftInfo.mlirSubtypes.size();
    mlir::Value result;

    for (int i = 0; i < numElements; ++i) {
        VarInfo& leftElemInfo = leftInfo.mlirSubtypes[i];
        VarInfo& rightElemInfo = rightInfo.mlirSubtypes[i];

        // Load the actual value from the memref
        mlir::Value leftElem = builder.create<mlir::memref::LoadOp>(loc, leftElemInfo.value, mlir::ValueRange{});
        mlir::Value rightElem = builder.create<mlir::memref::LoadOp>(loc, rightElemInfo.value, mlir::ValueRange{});

        mlir::Type elemType = leftElem.getType();

        mlir::Value elemEq;
        if (elemType.isa<mlir::TupleType>()) {
            elemEq = mlirTupleEquals(leftElemInfo, rightElemInfo, loc, builder); // recursive for nested tuples
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


void MLIRGen::visit(EqExpr* node){
    if (tryEmitConstantForNode(node)) return;
    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    mlir::Value left = leftInfo.value;
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    mlir::Value right = rightInfo.value;

    // If left/right are memref descriptors, load the contained scalar so the
    // equality helpers compare raw scalar types (integers/floats) rather than
    // memref descriptors.
    if (left.getType().isa<mlir::MemRefType>()) {
        left = builder_.create<mlir::memref::LoadOp>(loc_, leftInfo.value, mlir::ValueRange{});
    }
    if (right.getType().isa<mlir::MemRefType>()) {
        right = builder_.create<mlir::memref::LoadOp>(loc_, rightInfo.value, mlir::ValueRange{});
    }

    mlir::Type type = left.getType();
    mlir::Value result;

    if (type.isa<mlir::TupleType>()) {
        result = mlirTupleEquals(leftInfo, rightInfo, loc_, builder_);
    } else {
        result = mlirScalarEquals(left, right, loc_, builder_);
    }

    if (node->op == "!=") {
        auto one = builder_.create<mlir::arith::ConstantOp>(loc_, result.getType(), builder_.getIntegerAttr(result.getType(), 1));
        result = builder_.create<mlir::arith::XOrIOp>(loc_, result, one);
    }

    // Store the comparison result into a memref-backed VarInfo and push
    VarInfo outVar(node->type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}

void MLIRGen::visit(AndExpr* node){
    if (tryEmitConstantForNode(node)) return;
    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    mlir::Value left = leftInfo.value;
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    mlir::Value right = rightInfo.value;

    auto andOp = builder_.create<mlir::arith::AndIOp>(loc_, left, right);
    
    // Wrap boolean result into memref-backed VarInfo
    VarInfo outVar(leftInfo.type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, andOp, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}

void MLIRGen::visit(OrExpr* node){
    if (tryEmitConstantForNode(node)) return;
    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    mlir::Value left = leftInfo.value;
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    mlir::Value right = rightInfo.value;

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
