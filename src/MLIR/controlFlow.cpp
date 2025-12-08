#include "MLIRgen.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include <optional>

// Helper function to check if a block should be skipped (contains only a var declaration)
static bool shouldSkipVarDeclarationBlock(std::shared_ptr<BlockNode> block) {
    if (!block) return false;
    if (block->decs.size() != 1 || !block->stats.empty()) return false;
    
    auto dec = block->decs[0];
    if (auto typedDec = std::dynamic_pointer_cast<TypedDecNode>(dec)) {
        return typedDec->qualifier == "var";
    } else if (auto inferredDec = std::dynamic_pointer_cast<InferredDecNode>(dec)) {
        return inferredDec->qualifier == "var";
    } else if (auto tupleDec = std::dynamic_pointer_cast<TupleTypedDecNode>(dec)) {
        return tupleDec->qualifier == "var";
    }
    return false;
}

void MLIRGen::visit(IfNode* node) {
    // Evaluate the condition expression in the current block
    node->cond->accept(*this);
    VarInfo condVarInfo = popValue();
    
    // Normalize condition to SSA (load memref if needed)
    mlir::Value conditionValue = getSSAValue(condVarInfo);

    bool hasElse = (node->elseBlock != nullptr) || (node->elseStat != nullptr);

    // Current block and parent region where new blocks will be inserted.
    mlir::Block *currentBlock = builder_.getBlock();
    if (!currentBlock) {
        throw std::runtime_error("IfNode: builder has no current block");
    }
    mlir::Region *region = currentBlock->getParent();
    if (!region) {
        throw std::runtime_error("IfNode: current block has no parent region");
    }

    // Create then/else/merge blocks in the same region.
    auto *thenBlock = new mlir::Block();
    region->push_back(thenBlock);

    mlir::Block *elseBlock = nullptr;
    if (hasElse) {
        elseBlock = new mlir::Block();
        region->push_back(elseBlock);
    }

    auto *mergeBlock = new mlir::Block();
    region->push_back(mergeBlock);

    // Conditional branch from the current block.
    if (hasElse) {
        builder_.create<mlir::cf::CondBranchOp>(
            loc_, conditionValue,
            thenBlock, mlir::ValueRange{},
            elseBlock, mlir::ValueRange{});
    } else {
        builder_.create<mlir::cf::CondBranchOp>(
            loc_, conditionValue,
            thenBlock, mlir::ValueRange{},
            mergeBlock, mlir::ValueRange{});
    }

    // Then branch.
    builder_.setInsertionPointToStart(thenBlock);
    if (node->thenBlock) {
        // Skip var declarations in single-statement if blocks (they would be scoped and erased anyway)
        if (!shouldSkipVarDeclarationBlock(node->thenBlock)) {
            node->thenBlock->accept(*this);
        }
    } else if (node->thenStat) {
        node->thenStat->accept(*this);
    }
    if (thenBlock->empty() ||
        !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder_.setInsertionPointToEnd(thenBlock);
        builder_.create<mlir::cf::BranchOp>(loc_, mergeBlock);
    }

    // Else branch (if present).
    if (hasElse) {
        builder_.setInsertionPointToStart(elseBlock);
        if (node->elseBlock) {
            // Skip var declarations in single-statement if blocks (they would be scoped and erased anyway)
            if (!shouldSkipVarDeclarationBlock(node->elseBlock)) {
                node->elseBlock->accept(*this);
            }
        } else if (node->elseStat) {
            node->elseStat->accept(*this);
        }
        if (elseBlock->empty() ||
            !elseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            builder_.setInsertionPointToEnd(elseBlock);
            builder_.create<mlir::cf::BranchOp>(loc_, mergeBlock);
        }
    }

    // Continue inserting after the if in the merge block.
    builder_.setInsertionPointToStart(mergeBlock);
}

void MLIRGen::visit(LoopNode* node) {
    if (!node->body) {
        throw std::runtime_error("LoopNode: loop body is null");
    }

    mlir::Block *preBlock = builder_.getBlock();
    if (!preBlock) {
        throw std::runtime_error("LoopNode: builder has no current block");
    }
    mlir::Region *region = preBlock->getParent();
    if (!region) {
        throw std::runtime_error("LoopNode: current block has no parent region");
    }

    // Create blocks for the loop structure.
    auto *bodyBlock = new mlir::Block();
    region->push_back(bodyBlock);

    auto *exitBlock = new mlir::Block();
    region->push_back(exitBlock);

    mlir::Block *condBlock = nullptr;
    if (node->kind == LoopKind::While || node->kind == LoopKind::WhilePost) {
        condBlock = new mlir::Block();
        region->push_back(condBlock);
    }

    // Initial branch from the pre-loop block.
    if (node->kind == LoopKind::While) {
        builder_.create<mlir::cf::BranchOp>(loc_, condBlock);
    } else { // Plain and WhilePost: execute body first.
        builder_.create<mlir::cf::BranchOp>(loc_, bodyBlock);
    }

    // For pre-check while, emit the condition block now.
    if (node->kind == LoopKind::While && node->cond) {
        builder_.setInsertionPointToStart(condBlock);
        node->cond->accept(*this);
        VarInfo condVarInfo = popValue();
        mlir::Value condVal = getSSAValue(condVarInfo);
        builder_.create<mlir::cf::CondBranchOp>(
            loc_, condVal,
            bodyBlock, mlir::ValueRange{},
            exitBlock, mlir::ValueRange{});
    }

    // Enter loop body.
    builder_.setInsertionPointToStart(bodyBlock);
    LoopContext loopCtx;
    loopCtx.exitBlock = exitBlock;
    loopCtx.continueBlock =
        (node->kind == LoopKind::Plain ? bodyBlock : condBlock);
    loopContexts_.push_back(loopCtx);

    node->body->accept(*this);

    loopContexts_.pop_back();

    // After the body, if the current block is not yet terminated, branch to the appropriate continuation (condition or body) or emit the post condition for WhilePost.
    mlir::Block *curBlock = builder_.getBlock();
    auto needsTerminator = [&](mlir::Block *b) {
        return b && (b->empty() ||
                     !b->back().hasTrait<mlir::OpTrait::IsTerminator>());
    };

    if (node->kind == LoopKind::Plain) {
        if (needsTerminator(curBlock)) {
            builder_.create<mlir::cf::BranchOp>(loc_, bodyBlock);
        }
    } else if (node->kind == LoopKind::While) {
        if (needsTerminator(curBlock)) {
            builder_.create<mlir::cf::BranchOp>(loc_, condBlock);
        }
    } else if (node->kind == LoopKind::WhilePost) {
        if (!condBlock) {
            throw std::runtime_error(
                "LoopNode: WhilePost loop missing condition block");
        }
        if (needsTerminator(curBlock)) {
            builder_.create<mlir::cf::BranchOp>(loc_, condBlock);
        }
        builder_.setInsertionPointToStart(condBlock);
        if (!node->cond) {
            throw std::runtime_error(
                "LoopNode: WhilePost kind requires a condition");
        }
        node->cond->accept(*this);
        VarInfo condVarInfo = popValue();
        mlir::Value condVal = getSSAValue(condVarInfo);
        builder_.create<mlir::cf::CondBranchOp>(
            loc_, condVal,
            bodyBlock, mlir::ValueRange{},
            exitBlock, mlir::ValueRange{});
    }

    // Continue inserting after the loop in the exit block.
    builder_.setInsertionPointToStart(exitBlock);
}

void MLIRGen::visit(IteratorLoopNode* node) {
    if (!node->lowered) {
        throw std::runtime_error("IteratorLoopNode: missing lowered while-form; semantic pass should have populated it.");
    }
    node->lowered->accept(*this);
}

void MLIRGen::visit(GeneratorExprNode* node) {
    // Generators should have been lowered to allocation + fill loops.
    if (!node->lowered) {
        throw std::runtime_error("GeneratorExprNode: missing lowered form; semantic pass should have populated it.");
    }
    // Try to pre-compute runtime length(s) for dynamic range domains to allocate the result.
    auto computeLenFromRange = [&](std::shared_ptr<RangeExprNode> rangeDom) -> mlir::Value {
        auto getIntVal = [&](std::shared_ptr<ExprNode> expr, int fallback) -> mlir::Value {
            if (!expr) {
                auto cstTy = builder_.getI32Type();
                return builder_.create<mlir::arith::ConstantOp>(loc_, cstTy, builder_.getIntegerAttr(cstTy, fallback));
            }
            expr->accept(*this);
            VarInfo v = popValue();
            return getSSAValue(v); // load if memref-backed
        };
        mlir::Value startV = getIntVal(rangeDom->start, 1);
        mlir::Value endV   = getIntVal(rangeDom->end, 1);
        mlir::Value stepV  = getIntVal(rangeDom->step, 1);

        auto cstTy = builder_.getI32Type();
        mlir::Value zero = builder_.create<mlir::arith::ConstantOp>(loc_, cstTy, builder_.getIntegerAttr(cstTy, 0));
        mlir::Value one  = builder_.create<mlir::arith::ConstantOp>(loc_, cstTy, builder_.getIntegerAttr(cstTy, 1));

        // len = (end - start) / step + 1, but clamp to 0 if end < start
        mlir::Value diff    = builder_.create<mlir::arith::SubIOp>(loc_, endV, startV);
        mlir::Value div     = builder_.create<mlir::arith::DivSIOp>(loc_, diff, stepV);
        mlir::Value lenVal  = builder_.create<mlir::arith::AddIOp>(loc_, div, one);
        mlir::Value endLtStart = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::slt, endV, startV);
        lenVal = builder_.create<mlir::arith::SelectOp>(loc_, endLtStart, zero, lenVal);
        auto idxTy = builder_.getIndexType();
        return builder_.create<mlir::arith::IndexCastOp>(loc_, idxTy, lenVal);
    };

    std::vector<mlir::Value> dynLens;
    if (!node->type.dims.empty() && !node->domains.empty()) {
        if (node->type.dims[0] < 0) {
            if (auto rangeDom = std::dynamic_pointer_cast<RangeExprNode>(node->domains[0].second)) {
                dynLens.push_back(computeLenFromRange(rangeDom));
            }
        }
        if (node->type.dims.size() > 1 && node->type.dims[1] < 0 && node->domains.size() > 1) {
            if (auto rangeDom1 = std::dynamic_pointer_cast<RangeExprNode>(node->domains[1].second)) {
                dynLens.push_back(computeLenFromRange(rangeDom1));
            }
        }
    }

    // Manually emit lowered decs/stats in the current scope (avoid new scope)
    for (const auto &d : node->lowered->decs) {
        if (!d) continue;
        // Special-case the generated result array when dynamic length(s): allocate with computed sizes
        if (auto arrDec = std::dynamic_pointer_cast<ArrayTypedDecNode>(d)) {
            if (arrDec->id == node->loweredResultName) {
                VarInfo* var = currScope_->resolveVar(arrDec->id, arrDec->line);
                if (!var) throw std::runtime_error("GeneratorExprNode: result var not found");
                bool anyDynamic = false;
                for (int dim : var->type.dims) { if (dim < 0) { anyDynamic = true; break; } }
                if (anyDynamic && !dynLens.empty()) {
                    if (!var->value) {
                        allocaVar(var, arrDec->line, dynLens);
                    }
                    continue; // skip default visitor to avoid double alloc
                }
            }
        }
        d->accept(*this);
    }
    for (const auto &s : node->lowered->stats) {
        if (s) s->accept(*this);
    }
    // Push result VarInfo
    VarInfo* res = currScope_->resolveVar(node->loweredResultName, node->line);
    if (!res) {
        throw std::runtime_error("GeneratorExprNode: result variable '" + node->loweredResultName + "' not found");
    }
    pushValue(*res);
}