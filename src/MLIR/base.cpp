#include "CompileTimeExceptions.h"
#include "MLIRgen.h"

MLIRGen::MLIRGen(BackEnd& backend)
    : backend_(backend),
      builder_(*backend.getBuilder()),
      allocaBuilder_(*backend.getBuilder()),
      module_(backend.getModule()),
      context_(backend.getContext()),
      loc_(backend.getLoc()),
      root_(nullptr),
      currScope_(nullptr),
      scopeMap_(nullptr) {
}

MLIRGen::MLIRGen(BackEnd& backend, Scope* rootScope, const std::unordered_map<const ASTNode*, Scope*>* scopeMap)
    : backend_(backend),
      builder_(*backend.getBuilder()),
      allocaBuilder_(*backend.getBuilder()),
      module_(backend.getModule()),
      context_(backend.getContext()),
      loc_(backend.getLoc()),
      root_(rootScope),
      currScope_(nullptr),
      scopeMap_(scopeMap) {
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
    // Declaration for MathError function
    if (!module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("MathError")) {
        auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
        auto voidTy = mlir::LLVM::LLVMVoidType::get(&context_);
        auto mathErrorType = mlir::LLVM::LLVMFunctionType::get(voidTy, ptrTy, false);
        builder_.create<mlir::LLVM::LLVMFuncOp>(loc_, "MathError", mathErrorType);
    }
    createGlobalStringIfMissing("%d\0", "intFormat");
    createGlobalStringIfMissing("%c\0", "charFormat");
    createGlobalStringIfMissing("%g\0", "floatFormat");
    createGlobalStringIfMissing("%s\0", "strFormat");
    createGlobalStringIfMissing("\n\0", "newline");
}
void MLIRGen::visit(ArrayStrideExpr *node) {throw MainError(1, "Not yet implemented");}
void MLIRGen::visit(ArraySliceExpr *node) {throw MainError(1, "Not yet implemented");}
void MLIRGen::visit(ArrayAccessExpr *node) {throw MainError(1, "Not yet implemented");}
void MLIRGen::visit(ArrayInitNode *node) {throw MainError(1, "Not yet implemented");}
void MLIRGen::visit(ArrayDecNode *node) {throw MainError(1, "Not yet implemented");}
void MLIRGen::visit(ArrayTypeNode *node) {throw MainError(1, "Not yet implemented");}
void MLIRGen::visit(ExprListNode *node) {throw MainError(1, "Not yet implemented");}
void MLIRGen::visit(ArrayLiteralNode *node) {throw MainError(1, "Not yet implemented");}
void MLIRGen::visit(RangeExprNode *node) {throw MainError(1, "Not yet implemented");}
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
        auto globalTypedDec = std::dynamic_pointer_cast<TypedDecNode>(n);
        auto globalInferredDec = std::dynamic_pointer_cast<InferredDecNode>(n);
        auto globalTupleDec = std::dynamic_pointer_cast<TupleTypedDecNode>(n);

        if (!globalTypedDec && !globalInferredDec && !globalTupleDec) continue;

        CompleteType gType(BaseType::UNKNOWN);
        std::shared_ptr<ExprNode> initExpr;
        std::string qualifier;
        std::string name;

        if (globalTypedDec) {
            gType = globalTypedDec->type_alias ? globalTypedDec->type_alias->type : CompleteType(BaseType::UNKNOWN);
            qualifier = globalTypedDec->qualifier;
            initExpr = globalTypedDec->init;
            name = globalTypedDec->name;
        } else if (globalInferredDec) {
            gType = globalInferredDec->type;
            qualifier = globalInferredDec->qualifier;
            initExpr = globalInferredDec->init;
            name = globalInferredDec->name;
        } else if (globalTupleDec) {
            gType = globalTupleDec->type;
            qualifier = globalTupleDec->qualifier;
            initExpr = globalTupleDec->init;
            name = globalTupleDec->name;
        }

        if (qualifier == "var" || !initExpr) {
            throw std::runtime_error("FATAL: Var global or missing initializer for '" + name + "'.");
        }

        // Tuple globals - represent as a single LLVM struct global
        if (gType.baseType == BaseType::TUPLE) {
            auto tupleLit = std::dynamic_pointer_cast<TupleLiteralNode>(initExpr);
            if (!tupleLit) {
                throw std::runtime_error("Global tuple '" + name + "' missing initializer.");
            }
            if (gType.subTypes.size() != tupleLit->elements.size()) {
                throw std::runtime_error("Global tuple '" + name + "' len mismatch with literal.");
            }

            // Ensure each element is a constant we can lower.
            std::vector<mlir::Attribute> elemAttrs;
            elemAttrs.reserve(gType.subTypes.size());
            for (size_t i = 0; i < gType.subTypes.size(); ++i) {
                mlir::Attribute elemAttr =
                    extractConstantValue(tupleLit->elements[i], gType.subTypes[i]);
                if (!elemAttr) {
                    throw std::runtime_error(
                        "Global tuple '" + name +
                        "' has non-constant element at index " +
                        std::to_string(i) + ".");
                }
                elemAttrs.push_back(elemAttr);
            }

            // Create the LLVM struct global with an initializer region.
            mlir::Type structTy = getLLVMType(gType);
            auto global = moduleBuilder->create<mlir::LLVM::GlobalOp>(
                loc_, structTy, true, mlir::LLVM::Linkage::Internal, name, mlir::Attribute(), 0);

            mlir::Region &initRegion = global.getInitializerRegion();
            auto *block = new mlir::Block();
            initRegion.push_back(block);
            moduleBuilder->setInsertionPointToStart(block);

            mlir::Value structVal =
                moduleBuilder->create<mlir::LLVM::UndefOp>(loc_, structTy);
            for (size_t i = 0; i < gType.subTypes.size(); ++i) {
                mlir::Type elemTy = getLLVMType(gType.subTypes[i]);
                mlir::Attribute elemAttr = elemAttrs[i];
                mlir::Value elemConst =
                    moduleBuilder->create<mlir::LLVM::ConstantOp>(
                        loc_, elemTy, elemAttr);
                llvm::SmallVector<int64_t, 1> pos{
                    static_cast<int64_t>(i)};
                structVal = moduleBuilder->create<mlir::LLVM::InsertValueOp>(
                    loc_, structVal, elemConst, pos);
            }

            moduleBuilder->create<mlir::LLVM::ReturnOp>(loc_, structVal);
        } else {
            // Scalar global
            mlir::Attribute initAttr = extractConstantValue(initExpr, gType);
            if (!initAttr) {
                throw std::runtime_error("Missing constant initializer for global '" + name + "'.");
            }
            (void) createGlobalVariable(name, gType, /*isConst=*/true, initAttr);
        }
    }

    moduleBuilder->restoreInsertionPoint(savedIP);

    // Second pass: lower procedures/functions and their prototypes
    for (auto& n : node->stats) {
        if (std::dynamic_pointer_cast<ProcedurePrototypeNode>(n) ||
            std::dynamic_pointer_cast<ProcedureBlockNode>(n) ||
            std::dynamic_pointer_cast<FuncStatNode>(n) ||
            std::dynamic_pointer_cast<FuncBlockNode>(n) ||
            std::dynamic_pointer_cast<FuncPrototypeNode>(n)) {
            n->accept(*this);
        }
    }
}

void MLIRGen::visit(TypeCastNode* node) {
    node->expr->accept(*this);
    VarInfo from = popValue();
    VarInfo result = castType(&from, &node->type, node->line);
    pushValue(result);
}

void MLIRGen::visit(BlockNode* node) {
    // Enter the corresponding semantic child scope if present
    Scope* saved = currScope_;
    auto it = scopeMap_->find(node);
    if (it != scopeMap_->end()) {
        currScope_ = it->second;
    } else {
        throw std::runtime_error("FATAL: no corresponding scope found for BlockNode instance.");
    }

    // Process declarations
    for (const auto& d : node->decs) {
        if (d) {
            d->accept(*this);
        }
    }
    
    // After processing declarations, prevent further declarations in this block
    currScope_->disableDeclarations();
    
    // Process statements; stop once we hit a terminator in the current block.
    for (const auto& s : node->stats) {
        if (s) {
            mlir::Block *block = builder_.getBlock();
            if (block && !block->empty() &&
                block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
                break;
            }
            s->accept(*this);
        }
    }

    currScope_ = saved;
}
