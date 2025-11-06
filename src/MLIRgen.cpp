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


#include <stdexcept>

MLIRGen::MLIRGen(BackEnd& backend, Scope* rootScope)
    : backend_(backend),
      builder_(*backend.getBuilder()),
      allocaBuilder_(*backend.getBuilder()), // Dummy init, will be set in FileNode
      module_(backend.getModule()),
      context_(backend.getContext()),
      loc_(backend.getLoc()),
      root_(rootScope) {

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
    createGlobalStringIfMissing("\n\0", "newline");
}

MLIRGen::MLIRGen(BackEnd& backend)
// TODO: is this constructor needed?
    : backend_(backend),
      builder_(*backend.getBuilder()),
      allocaBuilder_(*backend.getBuilder()), // Dummy init, will be set in FileNode
      module_(backend.getModule()),
      context_(backend.getContext()),
      loc_(backend.getLoc()) {}


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
    // Start from the semantic root, then descend into the top-level global scope
    currScope_ = root_;
    if (currScope_ && !currScope_->children().empty()) {
        currScope_ = currScope_->children().front().get();
    }

    // Separate nodes by type: declarations, functions/procedures, and statements
    std::vector<std::shared_ptr<ASTNode>> declarations;
    std::vector<std::shared_ptr<ASTNode>> functions;
    std::vector<std::shared_ptr<ASTNode>> statements;
    
    for (auto& child : node->stats) {
        // Determine the type of node
        if (std::dynamic_pointer_cast<DecNode>(child)) {
            declarations.push_back(child);
        } else if (std::dynamic_pointer_cast<FuncStatNode>(child) ||
                   std::dynamic_pointer_cast<FuncBlockNode>(child) ||
                   std::dynamic_pointer_cast<FuncPrototypeNode>(child) ||
                   std::dynamic_pointer_cast<ProcedureNode>(child)) {
            functions.push_back(child);
        } else if (std::dynamic_pointer_cast<StatNode>(child)) {
            statements.push_back(child);
        } else if (std::dynamic_pointer_cast<TypeAliasDecNode>(child)) {
            // Type aliases are also declarations
            declarations.push_back(child);
        }
    }

    // Set insertion point to module level for creating globals
    auto* moduleBuilder = backend_.getBuilder().get();
    auto savedInsertionPoint = moduleBuilder->saveInsertionPoint();
    moduleBuilder->setInsertionPointToStart(module_.getBody());
    
    for (auto& decl : declarations) {
        decl->accept(*this);
    }
    
    // Process functions/procedures (emit as separate functions)
    // Functions will set their own insertion points
    for (auto& func : functions) {
        func->accept(*this);
    }
    
    // Process statements - wrap in main() function
    // Create main if there are statements OR deferred initializations
    if (!statements.empty() || !deferredInits_.empty()) {
        // Restore builder to module level to create main function
        moduleBuilder->restoreInsertionPoint(savedInsertionPoint);
        moduleBuilder->setInsertionPointToStart(module_.getBody());
        
        mlir::FunctionType mainType = builder_.getFunctionType({}, {builder_.getI32Type()});
        auto mainFunc = builder_.create<mlir::func::FuncOp>(loc_, "main", mainType);
        mlir::Block* entryBlock = mainFunc.addEntryBlock();
        
        // The allocaBuilder points to the very beginning of the function.
        // All memref.alloca operations MUST be generated here.
        allocaBuilder_ = mlir::OpBuilder(entryBlock, entryBlock->begin());

        // The main builder starts generating code inside the main function's entry block.
        builder_.setInsertionPointToStart(entryBlock);

        // Process deferred global variable initializations first
        // This ensures globals are initialized before they're used in statements
        for (auto& deferredInit : deferredInits_) {
            initializeGlobalInMain(deferredInit.varName, deferredInit.initExpr);
        }
        
        // Clear the deferred initializations list
        deferredInits_.clear();

        for (auto& stat : statements) {
            stat->accept(*this);
        }

        // Add a `return 0` at the end of main.
        // Ensure return is the last operation - find the last operation and insert after it
        if (!entryBlock->empty()) {
            // Remove any existing terminator
            mlir::Operation& lastOp = entryBlock->back();
            if (lastOp.hasTrait<mlir::OpTrait::IsTerminator>()) {
                lastOp.erase();
            }
            // Insert return after the last operation
            builder_.setInsertionPointAfter(&entryBlock->back());
        }
        mlir::Value zero = builder_.create<mlir::arith::ConstantOp>(
            loc_, builder_.getI32Type(), builder_.getI32IntegerAttr(0));
        builder_.create<mlir::func::ReturnOp>(loc_, zero);
    }
    
    // Restore the original insertion point
    moduleBuilder->restoreInsertionPoint(savedInsertionPoint);
}




// Functions
void MLIRGen::visit(FuncStatNode* node) { throw std::runtime_error("FuncStatNode not implemented"); }
void MLIRGen::visit(FuncPrototypeNode* node) { throw std::runtime_error("FuncPrototypeNode not implemented"); }
void MLIRGen::visit(FuncBlockNode* node) { throw std::runtime_error("FuncBlockNode not implemented"); }
void MLIRGen::visit(FuncCallExpr* node) { throw std::runtime_error("FuncCallExpr not implemented"); }
void MLIRGen::visit(ProcedureNode* node) { throw std::runtime_error("ProcedureNode not implemented"); }

// Declarations
mlir::Type MLIRGen::getLLVMType(CompleteType type) {
    switch (type.baseType) {
        case BaseType::BOOL:
            return builder_.getI1Type();
        case BaseType::CHARACTER:
            return builder_.getI8Type();
        case BaseType::INTEGER:
            return builder_.getI32Type();
        case BaseType::REAL:
            return builder_.getF32Type();
        case BaseType::TUPLE:
            // For tuples, we'll need to create a struct type or handle differently
            // For now, throw an error - tuples as globals need more complex handling
            throw std::runtime_error("Tuple types as globals not yet implemented");
        default:
            throw std::runtime_error("Unsupported type for global variable");
    }
}

mlir::Value MLIRGen::createGlobalVariable(const std::string& name, CompleteType type, bool isConst, mlir::Attribute initValue) {
    mlir::Type llvmType = getLLVMType(type);
    
    // Use module-level builder for globals (save current insertion point)
    auto* moduleBuilder = backend_.getBuilder().get();
    auto savedInsertionPoint = moduleBuilder->saveInsertionPoint();
    moduleBuilder->setInsertionPointToStart(module_.getBody());
    
    // Create the global variable at module level
    moduleBuilder->create<mlir::LLVM::GlobalOp>(
        loc_,
        llvmType,
        isConst,  // constant
        mlir::LLVM::Linkage::Internal,
        name,
        initValue,  // initial value (can be nullptr)
        0  // alignment (0 = default)
    );
    
    // Restore insertion point
    moduleBuilder->restoreInsertionPoint(savedInsertionPoint);
    
    // The global is now in the module - we'll access it via lookupSymbol when needed
    // Return a placeholder value - actual access will use AddressOfOp
    return nullptr;  // Actual access will be done via module_.lookupSymbol<LLVM::GlobalOp>(name)
}

mlir::Attribute MLIRGen::extractConstantValue(std::shared_ptr<ExprNode> expr, CompleteType targetType) {
    if (!expr) {
        return nullptr;
    }
    
    // Try to extract compile-time constant values from literal nodes
    if (auto intNode = std::dynamic_pointer_cast<IntNode>(expr)) {
        if (targetType.baseType == BaseType::INTEGER) {
            return builder_.getIntegerAttr(builder_.getI32Type(), intNode->value);
        } else if (targetType.baseType == BaseType::REAL) {
            // Integer can be promoted to real at compile time
            return builder_.getFloatAttr(builder_.getF32Type(), static_cast<float>(intNode->value));
        }
    } else if (auto realNode = std::dynamic_pointer_cast<RealNode>(expr)) {
        if (targetType.baseType == BaseType::REAL) {
            return builder_.getFloatAttr(builder_.getF32Type(), realNode->value);
        }
    } else if (auto charNode = std::dynamic_pointer_cast<CharNode>(expr)) {
        if (targetType.baseType == BaseType::CHARACTER) {
            return builder_.getIntegerAttr(builder_.getI8Type(), static_cast<int>(charNode->value));
        }
    } else if (auto trueNode = std::dynamic_pointer_cast<TrueNode>(expr)) {
        if (targetType.baseType == BaseType::BOOL) {
            return builder_.getIntegerAttr(builder_.getI1Type(), 1);
        }
    } else if (auto falseNode = std::dynamic_pointer_cast<FalseNode>(expr)) {
        if (targetType.baseType == BaseType::BOOL) {
            return builder_.getIntegerAttr(builder_.getI1Type(), 0);
        }
    }
    
    // Not a compile-time constant - return nullptr to indicate deferral needed
    return nullptr;
}

void MLIRGen::initializeGlobalInMain(const std::string& varName, std::shared_ptr<ExprNode> initExpr) {
    // Look up the global variable
    auto globalOp = module_.lookupSymbol<mlir::LLVM::GlobalOp>(varName);
    if (!globalOp) {
        throw SymbolError(1, "Global variable '" + varName + "' not found for initialization.");
    }
    
    // Get the address of the global
    mlir::Value globalAddr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalOp);
    
    // Evaluate the initialization expression
    initExpr->accept(*this);
    VarInfo initVarInfo = popValue();
    
    // Load the value from the memref
    mlir::Value initValue = builder_.create<mlir::memref::LoadOp>(
        loc_, initVarInfo.value, mlir::ValueRange{});
    
    // Store the value into the global using LLVM::StoreOp
    builder_.create<mlir::LLVM::StoreOp>(loc_, initValue, globalAddr);
}

void MLIRGen::visit(TypedDecNode* node) {
    // Resolve variable declared by semantic analysis
    VarInfo* declaredVar = currScope_->resolveVar(node->name);

    // Keep local constness consistent (semantic pass enforces this already)
    if (node->qualifier == "const") {
        declaredVar->isConst = true;
    } else if (node->qualifier == "var") {
        declaredVar->isConst = false;
    }

    // Ensure storage exists regardless of initializer
    if (!declaredVar->value && declaredVar->type.baseType != BaseType::TUPLE) {
        this->allocaVar(declaredVar);
    } else if (declaredVar->type.baseType == BaseType::TUPLE && declaredVar->mlirSubtypes.empty()) {
        this->allocaVar(declaredVar);
    }

    // Optional initializer: evaluate then assign with implicit promotion
    if (node->init) {
        node->init->accept(*this);
        VarInfo literal = popValue();
        this->assignTo(&literal, declaredVar);
    }
}

void MLIRGen::visit(InferredDecNode* node) {

    // 1. Analyze the initializer to determine type
    // 2. Create global with that type
    // 3. Defer initialization to main()
    
    // For now, this is a limitation - we need type inference at module level
    throw std::runtime_error("InferredDecNode as global not yet fully implemented - need type inference");
}

void MLIRGen::visit(TupleTypedDecNode* node) {
    // Tuples as globals are more complex - for now, throw an error
    // Proper implementation would create a struct type and handle element-wise initialization
    throw std::runtime_error("TupleTypedDecNode as global not yet fully implemented");
}
void MLIRGen::visit(TypeAliasDecNode* node) { throw std::runtime_error("TypeAliasDecNode not implemented"); }
void MLIRGen::visit(TypeAliasNode* node) { throw std::runtime_error("TypeAliasNode not implemented"); }
void MLIRGen::visit(TupleTypeAliasNode* node) { throw std::runtime_error("TupleTypeAliasNode not implemented"); }

// Statements
void MLIRGen::visit(AssignStatNode* node) { throw std::runtime_error("AssignStatNode not implemented"); }

void MLIRGen::visit(OutputStatNode* node) {
    
    if (!node->expr) {
        return;
    }
    
    // Evaluate the expression to get the value to print
    node->expr->accept(*this);
    VarInfo exprVarInfo = popValue();

    // Load the value from its memref
    mlir::Value loadedValue = builder_.create<mlir::memref::LoadOp>(
        loc_, exprVarInfo.value, mlir::ValueRange{});

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
void MLIRGen::visit(BreakStatNode* node) { throw std::runtime_error("BreakStatNode not implemented"); }
void MLIRGen::visit(ContinueStatNode* node) { throw std::runtime_error("ContinueStatNode not implemented"); }
void MLIRGen::visit(ReturnStatNode* node) { throw std::runtime_error("ReturnStatNode not implemented"); }
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
        
        // Remove any existing terminators
        int terminatorCount = 0;
        while (!thenBlock.empty() && thenBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            thenBlock.back().erase();
            terminatorCount++;
        }
        if (terminatorCount > 0) {
        }
        
        // Insert yield at the end of the block
        if (!thenBlock.empty()) {
            builder_.setInsertionPointAfter(&thenBlock.back());
        } else {
            builder_.setInsertionPointToStart(&thenBlock);
        }
        builder_.create<mlir::scf::YieldOp>(loc_);
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
        
        // Remove any existing terminators
        while (!elseBlock.empty() && elseBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            elseBlock.back().erase();
        }
        
        // Insert yield at the end of the block
        if (!elseBlock.empty()) {
            builder_.setInsertionPointAfter(&elseBlock.back());
        } else {
            builder_.setInsertionPointToStart(&elseBlock);
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
                mlirSubtype.isConst = true;
                varInfo->mlirSubtypes.emplace_back(mlirSubtype);
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

void MLIRGen::allocaVar(VarInfo* varInfo) {
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
        case BaseType::TUPLE: {
            if (varInfo->type.subTypes.size() < 2) {
                throw SizeError(1, "Error: Tuple must have at least 2 elements.");
            }
            for (CompleteType& subtype : varInfo->type.subTypes) {
                VarInfo mlirSubtype = VarInfo(subtype);
                mlirSubtype.isConst = varInfo->isConst;
                varInfo->mlirSubtypes.emplace_back(mlirSubtype);
                allocaVar(&varInfo->mlirSubtypes.back());
            }
            break;
        }
        default:
            throw std::runtime_error("allocaVar FATAL: unsupported type " +
                std::to_string(static_cast<int>(varInfo->type.baseType)));
    }
}

VarInfo MLIRGen::promoteType(VarInfo* from, CompleteType* toType) {
    if (from->type == *toType) {
        return *from; // no-op
    }
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

void MLIRGen::assignTo(VarInfo* literal, VarInfo* variable) {
    // Tuple assignment: element-wise
    if (variable->type.baseType == BaseType::TUPLE) {
        if (literal->type.baseType != BaseType::TUPLE) {
            throw AssignError(1, "Cannot assign non-tuple to tuple variable.");
        }
        if (variable->mlirSubtypes.empty()) {
            allocaVar(variable);
        }
        if (literal->mlirSubtypes.empty()) {
            throw AssignError(1, "Tuple source has no element storage.");
        }
        if (literal->type.subTypes.size() != variable->type.subTypes.size()) {
            throw AssignError(1, "Tuple arity mismatch in assignment.");
        }
        for (size_t i = 0; i < variable->mlirSubtypes.size(); ++i) {
            VarInfo srcElem = literal->mlirSubtypes[i];
            VarInfo& dstElem = variable->mlirSubtypes[i];
            VarInfo promoted = promoteType(&srcElem, &dstElem.type);
            mlir::Value loaded = builder_.create<mlir::memref::LoadOp>(loc_, promoted.value, mlir::ValueRange{});
            builder_.create<mlir::memref::StoreOp>(loc_, loaded, dstElem.value, mlir::ValueRange{});
        }
        return;
    }

    // Scalar assignment
    if (!variable->value) {
        allocaVar(variable);
    }
    VarInfo promoted = promoteType(literal, &variable->type);
    mlir::Value loadedVal = builder_.create<mlir::memref::LoadOp>(
        loc_, promoted.value, mlir::ValueRange{}
    );
    builder_.create<mlir::memref::StoreOp>(
        loc_, loadedVal, variable->value, mlir::ValueRange{}
    );
}


void MLIRGen::visit(ParenExpr* node) {
    node->expr->accept(*this);
}

void MLIRGen::visit(UnaryExpr* node) {
    node->operand->accept(*this);
    VarInfo operand = popValue();
    mlir::Value operandVal = operand.value;

    if(node->op == "-"){
        auto zero = builder_.create<mlir::arith::ConstantOp>(
            loc_, operandVal.getType(), builder_.getZeroAttr(operandVal.getType()));
        operand.value = builder_.create<mlir::arith::SubIOp>(loc_, zero, operandVal);
    }

    operand.identifier = "";
    pushValue(operand);
}

void MLIRGen::visit(ExpExpr* node) {
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

    //assume both operands are of same type
    //the left operand object is pushed back to the stack with a new value
    left.identifier = ""; 
    left.value = result;
    pushValue(left);
}

void MLIRGen::visit(MultExpr* node){
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

    leftInfo.identifier = "";
    leftInfo.value = result;
    pushValue(leftInfo);
}

void MLIRGen::visit(AddExpr* node){
    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    mlir::Value left = leftInfo.value;
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    mlir::Value right = rightInfo.value;

    mlir::Value result;
    if(left.getType().isa<mlir::IntegerType>()) {
        if (node->op == "+") {
            result = builder_.create<mlir::arith::AddIOp>(loc_, left, right);
        } else if (node->op == "-") {
            result = builder_.create<mlir::arith::SubIOp>(loc_, left, right);
        }
    } else if(left.getType().isa<mlir::FloatType>()) {
        if (node->op == "+") {
            result = builder_.create<mlir::arith::AddFOp>(loc_, left, right);
        } else if (node->op == "-") {
            result = builder_.create<mlir::arith::SubFOp>(loc_, left, right);
        }
    } else {
        throw std::runtime_error("MLIRGen Error: Unsupported type for addition.");
    }

    leftInfo.identifier = "";
    leftInfo.value = result;
    pushValue(leftInfo);
}

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


void MLIRGen::visit(NotExpr* node) {
    node->operand->accept(*this);
    VarInfo operandInfo = popValue();
    mlir::Value operand = operandInfo.value;

    auto one = builder_.create<mlir::arith::ConstantOp>(
        loc_, operand.getType(), builder_.getIntegerAttr(operand.getType(), 1));
    auto notOp = builder_.create<mlir::arith::XOrIOp>(loc_, operand, one);
    
    operandInfo.identifier = "";
    operandInfo.value = notOp;
    pushValue(operandInfo);
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
    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    mlir::Value left = leftInfo.value;
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    mlir::Value right = rightInfo.value;

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

    leftInfo.identifier = "";
    leftInfo.value = result;
    pushValue(leftInfo);
}

void MLIRGen::visit(AndExpr* node){
    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    mlir::Value left = leftInfo.value;
    node->right->accept(*this);
    VarInfo rightInfo = popValue();
    mlir::Value right = rightInfo.value;

    auto andOp = builder_.create<mlir::arith::AndIOp>(loc_, left, right);
    
    leftInfo.identifier = "";
    leftInfo.value = andOp;
    pushValue(leftInfo);
}

void MLIRGen::visit(OrExpr* node){
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

    leftInfo.identifier = "";
    leftInfo.value = result;
    pushValue(leftInfo);
}