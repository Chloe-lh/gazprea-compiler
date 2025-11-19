#include "MLIRgen.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"



void MLIRGen::visit(IfNode* node) {
    // Evaluate the condition expression in the current block
    node->cond->accept(*this);
    VarInfo condVarInfo = popValue();
    
    // Load the condition value from its memref in the current block
    mlir::Value conditionValue = builder_.create<mlir::memref::LoadOp>(
        loc_, condVarInfo.value, mlir::ValueRange{});

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
        node->thenBlock->accept(*this);
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
            node->elseBlock->accept(*this);
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
        mlir::Value condVal = builder_.create<mlir::memref::LoadOp>(
            loc_, condVarInfo.value, mlir::ValueRange{});
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
        mlir::Value condVal = builder_.create<mlir::memref::LoadOp>(
            loc_, condVarInfo.value, mlir::ValueRange{});
        builder_.create<mlir::cf::CondBranchOp>(
            loc_, condVal,
            bodyBlock, mlir::ValueRange{},
            exitBlock, mlir::ValueRange{});
    }

    // Continue inserting after the loop in the exit block.
    builder_.setInsertionPointToStart(exitBlock);
}