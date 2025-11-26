#include <any>
#include "AST.h"
#include "ASTBuilder.h"
#include "ASTBuilderUtils.h"

namespace gazprea{
// if: IF PARENLEFT expr PARENRIGHT (block|stat) (ELSE (block|stat))?;
std::any ASTBuilder::visitIfStat(gazprea::GazpreaParser::IfStatContext *ctx) {
  auto ifCtx = ctx->if_stat();
  if (!ifCtx) {
    return nullptr;
  }

  // Visit the condition and create the IfNode using the new constructor.
  auto cond = safe_any_cast_ptr<ExprNode>(visit(ifCtx->expr()));
  auto node = std::make_shared<IfNode>(cond);
  setLocationFromCtx(node, ctx);

  // The ANTLR grammar allows block, stat, or dec in the 'then' branch.
  if (!ifCtx->block().empty()) {
    node->thenBlock = safe_any_cast_ptr<BlockNode>(visit(ifCtx->block(0)));
  } else if (!ifCtx->dec().empty()) {
    // Wrap declaration in a BlockNode
    auto decAny = visit(ifCtx->dec(0));
    auto dec = safe_any_cast_ptr<DecNode>(decAny);
    if (dec) {
      std::vector<std::shared_ptr<DecNode>> decs;
      decs.push_back(dec);
      std::vector<std::shared_ptr<StatNode>> stats;
      node->thenBlock =
          std::make_shared<BlockNode>(std::move(decs), std::move(stats));
    }
  } else if (!ifCtx->stat().empty()) {
    node->thenStat = safe_any_cast_ptr<StatNode>(visit(ifCtx->stat(0)));
  }

  // Determine and visit the 'else' branch, if it exists.
  if (ifCtx->ELSE()) {
    bool thenWasBlock = (node->thenBlock != nullptr);

    if (thenWasBlock) {
      // If 'then' was a block, 'else' can be the second block, first
      // declaration, or first statement.
      if (ifCtx->block().size() > 1) {
        node->elseBlock = safe_any_cast_ptr<BlockNode>(visit(ifCtx->block(1)));
      } else if (!ifCtx->dec().empty()) {
        // Wrap declaration in a BlockNode
        auto decAny = visit(ifCtx->dec(0));
        auto dec = safe_any_cast_ptr<DecNode>(decAny);
        if (dec) {
          std::vector<std::shared_ptr<DecNode>> decs;
          decs.push_back(dec);
          std::vector<std::shared_ptr<StatNode>> stats;
          node->elseBlock =
              std::make_shared<BlockNode>(std::move(decs), std::move(stats));
        }
      } else if (!ifCtx->stat().empty()) {
        node->elseStat = safe_any_cast_ptr<StatNode>(visit(ifCtx->stat(0)));
      }
    } else { // 'then' was a statement
      // If 'then' was a statement, 'else' can be the first block, first
      // declaration, or second statement.
      if (!ifCtx->block().empty()) {
        node->elseBlock = safe_any_cast_ptr<BlockNode>(visit(ifCtx->block(0)));
      } else if (ifCtx->stat().size() > 1) {
        node->elseStat = safe_any_cast_ptr<StatNode>(visit(ifCtx->stat(1)));
      }
    }
  } 

  return node_any(std::move(node));
}

// LOOP (block|stat) (WHILE PARENLEFT expr PARENRIGHT END)? #LoopDefault
std::any ASTBuilder::visitLoopDefault(GazpreaParser::LoopDefaultContext *ctx) {
  std::shared_ptr<BlockNode> body = nullptr;
  std::shared_ptr<ExprNode> cond = nullptr;
  bool hasCond = false;

  // extract body if present
  if (ctx->block()) {
    auto anyBody = visit(ctx->block());
    if (anyBody.has_value()) {
      body = safe_any_cast_ptr<BlockNode>(anyBody);
    }
  }

  // extract optional condition expression (for while-like loops)
  if (ctx->expr()) {
    hasCond = true;
    auto anyCond = visit(ctx->expr());
    if (anyCond.has_value()) {
      cond = safe_any_cast_ptr<ExprNode>(anyCond);
    }
  }
  // Construct LoopNode with body and optional condition, then set the kind
  auto node = std::make_shared<LoopNode>(std::move(body), std::move(cond));
  setLocationFromCtx(node, ctx);
  if (hasCond) {
    node->kind = LoopKind::WhilePost; // body then condition (do-while style)
  } else {
    node->kind = LoopKind::Plain; // infinite loop / no condition
  }

  return node_any(std::move(node));
}

std::any ASTBuilder::visitLoopStat(GazpreaParser::LoopStatContext *ctx) {

  // TODO: fix this
  auto loopCtx = ctx->loop_stat();
  if (!loopCtx) {
    return nullptr;
  }
  return visit(loopCtx);
}

// LOOP (WHILE PARENLEFT expr PARENRIGHT) (block|stat) #WhileLoopBlock
std::any
ASTBuilder::visitWhileLoopBlock(GazpreaParser::WhileLoopBlockContext *ctx) {
  std::shared_ptr<ExprNode> init = nullptr;
  if (ctx->expr()) {
    auto exprAny = visit(ctx->expr());
    if (exprAny.has_value()) {
      init = safe_any_cast_ptr<ExprNode>(exprAny);
    }
  }
  std::shared_ptr<BlockNode> block = nullptr;
  std::shared_ptr<StatNode> stat = nullptr;
  // If the grammar produced a block use it, otherwise if it's a single
  // statement wrap that statement into a temporary BlockNode so LoopNode
  // always carries a BlockNode body.
  if (ctx->block()) {
    auto blockAny = visit(ctx->block());
    if (blockAny.has_value()) {
      block = safe_any_cast_ptr<BlockNode>(blockAny);
    }
  } else if (ctx->stat()) {
    auto statAny = visit(ctx->stat());
    if (statAny.has_value()) {
      stat = safe_any_cast_ptr<StatNode>(statAny);
    }
    if (stat) { // turn stats to body (blockNode)
      std::vector<std::shared_ptr<DecNode>> emptyDecs;
      std::vector<std::shared_ptr<StatNode>> stats;
      stats.push_back(stat);
      block =
          std::make_shared<BlockNode>(std::move(emptyDecs), std::move(stats));
    }
  }

  // Construct LoopNode: body (BlockNode) and condition (expr). Mark as
  // pre-check while.
  auto node = std::make_shared<LoopNode>(std::move(block), std::move(init));
  setLocationFromCtx(node, ctx);
  node->kind = LoopKind::While;

  return node_any(std::move(node));
}
}