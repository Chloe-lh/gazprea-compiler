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
    currScope_ = root_;
    
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
    // Get the variable's type from the type alias
    CompleteType varType = node->type_alias->type;
    
    // Check if variable is already declared in the current scope
    // We check the symbols map directly to avoid throwing if the variable doesn't exist
    const auto& symbols = currScope_->symbols();
    if (symbols.find(node->name) != symbols.end()) {
        throw SymbolError(1, "Variable '" + node->name + "' already declared.");
    }
    
    // Determine if this is a constant
    bool isConst = (node->qualifier == "const" || node->qualifier.empty());
    
    // Try to extract compile-time constant value from initializer
    mlir::Attribute initAttr = nullptr;
    if (node->init) {
        initAttr = extractConstantValue(node->init, varType);
    }
    
    // Create global variable with initializer (if compile-time constant)
    createGlobalVariable(
        node->name,
        varType,
        isConst,
        initAttr  // nullptr if not a compile-time constant
    );
    
    // Store in scope
    currScope_->declareVar(node->name, varType, isConst);
    
    // If initialization expression is not a compile-time constant, defer to main()
    if (node->init && !initAttr) {
        deferredInits_.push_back({node->name, node->init});
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
