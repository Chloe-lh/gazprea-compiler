#include "ASTBuilder.h"
#include "AST.h"
#include "ASTBuilderUtils.h"
#include "Types.h"
#include "antlr4-runtime.h"
#include <any>
#include <memory>
#include <stdexcept>
#include "CompileTimeExceptions.h"

using namespace gazprea;
using namespace antlr4;

/*
      //  DONE std::any visitFile(GazpreaParser::FileContext *ctx) override; 
       // std::any visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx) override;
     //   std::any visitFunctionBlockTupleReturn(GazpreaParser::FunctionBlockTupleReturnContext *ctx) override;
    //    std::any visitFunctionStat(GazpreaParser::FunctionStatContext *ctx) override;
        // std::any visitFunctionPrototype(GazpreaParser::FunctionPrototypeContext *ctx) override;
    //    std::any visitFunctionPrototypeTupleReturn(GazpreaParser::FunctionPrototypeTupleReturnContext *ctx) override;
        std::any visitProcedure(GazpreaParser::ProcedureContext *ctx) override;
        std::any visitExplicitTypedDec(GazpreaParser::ExplicitTypedDecContext *ctx) override;
        std::any visitInferredTypeDec(GazpreaParser::InferredTypeDecContext *ctx) override;
        std::any visitTupleTypedDec(GazpreaParser::TupleTypedDecContext *ctx) override;
        std::any visitAssignStat(GazpreaParser::AssignStatContext *ctx) override;
        std::any visitOutputStat(GazpreaParser::OutputStatContext *ctx) override;
        std::any visitInputStat(GazpreaParser::InputStatContext *ctx) override;
        std::any visitBreakStat(GazpreaParser::BreakStatContext *ctx) override;
        std::any visitContinueStat(GazpreaParser::ContinueStatContext *ctx) override;
        std::any visitReturnStat(GazpreaParser::ReturnStatContext *ctx) override;
        std::any visitCallStat(GazpreaParser::CallStatContext *ctx) override;
       // std::any visitType(GazpreaParser::TypeContext *ctx) override;
        std::any visitBasicTypeAlias(GazpreaParser::BasicTypeAliasContext *ctx) override;
        std::any visitTupleTypeAlias(GazpreaParser::TupleTypeAliasContext *ctx) override;
        std::any visitAndExpr(GazpreaParser::AndExprContext *ctx) override;
      //  std::any visitTrueExpr(GazpreaParser::TrueExprContext *ctx) override;
       // DONE std::any visitIdExpr(GazpreaParser::IdExprContext *ctx) override;
        std::any visitMultExpr(GazpreaParser::MultExprContext *ctx) override;
        std::any visitAddExpr(GazpreaParser::AddExprContext *ctx) override;
        std::any visitCompExpr(GazpreaParser::CompExprContext *ctx) override;
        std::any visitExpExpr(GazpreaParser::ExpExprContext *ctx) override;
        std::any visitUnaryExpr(GazpreaParser::UnaryExprContext *ctx) override;
        std::any visitTupleTypeCastExpr(GazpreaParser::TupleTypeCastExprContext *ctx) override;
        std::any visitOrExpr(GazpreaParser::OrExprContext *ctx) override;
     //   std::any visitFalseExpr(GazpreaParser::FalseExprContext *ctx) override;
     //   DONE std::any visitCharExpr(GazpreaParser::CharExprContext *ctx) override;
        std::any visitTupleAccessExpr(GazpreaParser::TupleAccessExprContext *ctx) override;
        std::any visitTupleLitExpr(GazpreaParser::TupleLitExprContext *ctx) override;
        std::any visitEqExpr(GazpreaParser::EqExprContext *ctx) override;
        std::any visitNotExpr(GazpreaParser::NotExprContext *ctx) override;
     //   DONE ::any visitIntExpr(GazpreaParser::IntExprContext *ctx) override;
     //   std::any visitParenExpr(GazpreaParser::ParenExprContext *ctx) override;
     //   DONE std::any visitRealExpr(GazpreaParser::RealExprContext *ctx) override;
        std::any visitTypeCastExpr(GazpreaParser::TypeCastExprContext *ctx) override;
        std::any visitFuncCallExpr(GazpreaParser::FuncCallExprContext *ctx) override;
        // std::any visitTuple_dec(GazpreaParser::Tuple_decContext *ctx) override;
        // std::any visitTuple_literal(GazpreaParser::Tuple_literalContext *ctx) override;
        std::any visitTuple_access(GazpreaParser::Tuple_accessContext *ctx) override;
      //  DONE std::any visitBlock(GazpreaParser::BlockContext *ctx) override;
        std::any visitIf(GazpreaParser::IfContext *ctx) override;
        std::any visitLoopDefault(GazpreaParser::LoopDefaultContext *ctx) override;
        std::any visitWhileLoopBlock(GazpreaParser::WhileLoopBlockContext *ctx) override;
        std::any visitQualifier(GazpreaParser::QualifierContext *ctx) override;
     //   DONE std::any visitReal(GazpreaParser::RealContext *ctx) override;
*/
std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
    std::vector<std::shared_ptr<ASTNode>> nodes;
    for (auto child: ctx->children) { 
        auto anyNode = visit(child);
        if (anyNode.has_value()){
            auto node = std::any_cast<std::shared_ptr<ASTNode>>(anyNode);
            if(node) nodes.push_back(node);
        }   
    }
    return std::make_shared<FileNode>(std::move(nodes));
}
std::any ASTBuilder::visitBlock(GazpreaParser::BlockContext *ctx){
    std::vector<std::shared_ptr<DecNode>> decs;
    std::vector<std::shared_ptr<StatNode>> stats;
    for(auto decCtx: ctx->dec()){
        auto decAny = visit(decCtx);
        auto dec = std::any_cast<std::shared_ptr<DecNode>>(decAny);
        if(dec) decs.push_back(dec);
    }
    for(auto statCtx: ctx->stat()){
        auto statAny = visit(statCtx);
        auto stat = std::any_cast<std::shared_ptr<StatNode>>(statAny);
        if(stat) stats.push_back(stat);
    }
    return std::make_shared<BlockNode>(std::move(decs), std::move(stats));
}
// returns CompleteType object based on grammar else returns unknown
std::any ASTBuilder::visitType(GazpreaParser::TypeContext *ctx){
    if(ctx->BOOLEAN()) return CompleteType(BaseType::BOOL);
    if(ctx->ID()) return CompleteType(BaseType::STRING);
    if(ctx->INTEGER()) return CompleteType(BaseType::INTEGER);
    if(ctx->REAL()) return CompleteType(BaseType::REAL);
    if(ctx->CHARACTER()) return CompleteType(BaseType::CHARACTER);
    return CompleteType(BaseType::UNKNOWN);
}

// since there is no function block node, return a Function Prototype with a body
// combines a function signature with a function body
std::any ASTBuilder::visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx){
    std::string funcName = ctx->ID(0)->getText();
    std::vector<std::pair<CompleteType, std::string>> params = gazprea::builder_utils::ExtractParams(*this, ctx);
    std::shared_ptr<BlockNode> body = nullptr;
    CompleteType returnType = gazprea::builder_utils::ExtractReturnType(*this, ctx);
    
    if(ctx->block()){ // function has a block
        auto anyBody = visit(ctx->block());
        if(anyBody.has_value() && anyBody.type() == typeid(std::shared_ptr<BlockNode>)){
             body = std::any_cast<std::shared_ptr<BlockNode>>(anyBody);
        }
    }
    auto node = std::make_shared<FuncPrototypeNode>(funcName, params, returnType);
    node->body = body;//assign superclass body
    return node;
}
// combines a function signature with a function body
std::any ASTBuilder::visitFunctionBlockTupleReturn(GazpreaParser::FunctionBlockTupleReturnContext *ctx){
    std::string funcName = ctx->ID(0)->getText();
    std::vector<std::pair<CompleteType, std::string>> params = gazprea::builder_utils::ExtractParams(*this, ctx);
    std::shared_ptr<BlockNode> body = nullptr;
    CompleteType returnType = gazprea::builder_utils::ExtractReturnType(*this, ctx);
    
    if(ctx->block()){ // function has a block
        auto anyBody = visit(ctx->block());
        if(anyBody.has_value() && anyBody.type() == typeid(std::shared_ptr<BlockNode>)){
             body = std::any_cast<std::shared_ptr<BlockNode>>(anyBody);
        }
    }
    auto node = std::make_shared<FuncPrototypeNode>(funcName, params, returnType);
    node->body = body;
    return node;
}
std::any ASTBuilder::visitFunctionPrototype(GazpreaParser::FunctionPrototypeContext *ctx){
    std::string funcName = ctx->ID(0)->getText();
    std::vector<std::pair<CompleteType, std::string>> params = gazprea::builder_utils::ExtractParams(*this, ctx);
    CompleteType returnType = gazprea::builder_utils::ExtractReturnType(*this, ctx);
    auto node = std::make_shared<FuncPrototypeNode>(funcName, params, returnType);
    // no body for a prototype
    return node;
}
std::any ASTBuilder::visitFunctionPrototypeTupleReturn(GazpreaParser::FunctionPrototypeTupleReturnContext *ctx){
    std::string funcName = ctx->ID(0)->getText();
    std::vector<std::pair<CompleteType, std::string>> params = gazprea::builder_utils::ExtractParams(*this, ctx);
    CompleteType returnType = gazprea::builder_utils::ExtractReturnType(*this, ctx);
    // no body for a prototype
    auto node = std::make_shared<FuncPrototypeNode>(funcName, params, returnType);
    return node;
}
std::any ASTBuilder::visitFunctionStat(GazpreaParser::FunctionStatContext *ctx){
    std::string funcName = ctx->ID(0)->getText();
    std::vector<std::pair<CompleteType, std::string>> params = gazprea::builder_utils::ExtractParams(*this, ctx);
    CompleteType returnType = gazprea::builder_utils::ExtractReturnType(*this, ctx);
    std::shared_ptr<StatNode> returnStat = nullptr;
    if (ctx->stat()) {
        auto anyStat = visit(ctx->stat());
        if (anyStat.has_value()) {
            // Preferred: safe-check the type then any_cast
            if (anyStat.type() == typeid(std::shared_ptr<StatNode>)) {
                returnStat = std::any_cast<std::shared_ptr<StatNode>>(anyStat);
            } else {
                // fallback attempt: try catching bad_any_cast to avoid crashes
                try {
                    returnStat = std::any_cast<std::shared_ptr<StatNode>>(anyStat);
                } catch (const std::bad_any_cast&) {
                    // not a StatNode — handle gracefully, e.g. leave nullptr or log
                }
            }
        }
    }
    // no body for a prototype
    auto node = std::make_shared<FuncStatNode>(funcName, params, returnType, returnStat);
    return node;
}

std::any ASTBuilder::visitIntExpr(GazpreaParser::IntExprContext *ctx){
    try{
        std::string lit = ctx->INT()->getText();
        size_t parsed = 0;
        int64_t value64 = std::stoll(lit, &parsed, 10);
        if (parsed != lit.size()) {
            throw LiteralError(ctx->getStart()->getLine(), "integer literal exceeds 32 bits");
        }
        if(value64 < std::numeric_limits<int32_t>::min()||
            value64 > std::numeric_limits<int32_t>::max()){
                throw LiteralError(ctx->getStart()->getLine(), "integer literal exceeds 32 bits");
        }
        int value32 = static_cast<int>(value64);
        auto node = std::make_shared<IntNode>(value32);
        node->type = CompleteType(BaseType::INTEGER);
        return node;
    }catch(const std::out_of_range&){
        throw LiteralError(ctx->getStart()->getLine(), "integer literal out of bounds");
    }catch(const std::invalid_argument&){
        throw LiteralError(ctx->getStart()->getLine(), "invalid integer literal");
    }
}
std::any ASTBuilder::visitIdExpr(GazpreaParser::IdExprContext *ctx){
    std::string name = ctx->ID()->getText();
    // Create the IdNode and return it. Don't assign a concrete type here —
    // identifier types are resolved in the name-resolution / type-resolution pass.
    auto node = std::make_shared<IdNode>(name);
    node->type = CompleteType(BaseType::STRING);
    return node;
}
std::any ASTBuilder::visitCharExpr(GazpreaParser::CharExprContext *ctx){
    std::string text = ctx->getText();
    char value;
    // 'c'
    if (text.length() >= 3 && text[0] == '\'' && text.back() == '\'') {
        // remove ticks
        std::string sub = text.substr(1, text.length()-2);
        if(sub.length()==1){
            value = sub[0];
        }else if(sub[0]== '\\'){ // '\\' is one char
            switch (sub[1]){ // gets the next char
                case '0': value = '\0'; break; //null
                case 'a': value = '\a'; break; //bell
                case 'b': value = '\b'; break; //backspace
                case 't': value = '\t'; break; //tab
                case 'n': value = '\n'; break; //line feed
                case 'r': value = '\r'; break; //carriage return
                case '"': value = '\"'; break; //quotation mark
                case '\'': value = '\''; break; //apostrophe
                case '\\': value = '\\'; break; //backslash
                default: value = sub[1];
            }
        }else{
            value = sub[0];
        }
    }else{ //invalid character
        throw LiteralError(ctx->getStart()->getLine(), "invalid character");
    }
    auto node = std::make_shared<CharNode>(value);
    node->type = CompleteType(BaseType::CHARACTER);
    return node;
}

std::any ASTBuilder::visitRealExpr(GazpreaParser::RealExprContext *ctx){
    std::string text = ctx->real()->getText();
    // apply leading zero
    double value;
    if (!text.empty() && text[0] == '.'){
        text = "0" + text;
    } else if (text.size() >= 2 && text[0] == '-' && text[1] == '.'){
        text = "-0" + text.substr(1);
    }
    try{
        value = std::stod(text); //convert to real
    }catch(const std::out_of_range&){
        throw LiteralError(ctx->getStart()->getLine(), "real literal out of bounds");
    }catch(const std::invalid_argument&){
        throw LiteralError(ctx->getStart()->getLine(), "invalid real literal");
    }
    auto node = std::make_shared<RealNode>(value);
    node->type = CompleteType(BaseType::REAL);
    return node;
}

std::any ASTBuilder::visitTrueExpr(GazpreaParser::TrueExprContext *ctx){
    auto node = std::make_shared<TrueNode>();
    node->type = CompleteType(BaseType::BOOL);
    return node;
}
std::any ASTBuilder::visitFalseExpr(GazpreaParser::FalseExprContext *ctx){
    auto node = std::make_shared<FalseNode>();
    node->type = CompleteType(BaseType::BOOL);
    return node;
}

std::any ASTBuilder::visitTuple_literal(GazpreaParser::Tuple_literalContext *ctx){
    std::vector<std::shared_ptr<ExprNode>> elements;
    for(auto exprCtx: ctx->expr()){
        auto exprAny = visit(exprCtx);
        auto expr = std::any_cast<std::shared_ptr<ExprNode>>(exprAny);
        if(expr) elements.push_back(expr);
    }
    auto node = std::make_shared<TupleLiteralNode>(elements);

    // Build the tuple CompleteType from the element expression types so
    // downstream passes have subtype information available.
    std::vector<CompleteType> elemTypes;
    elemTypes.reserve(elements.size());
    for (auto &el : elements) {
        if (el) elemTypes.push_back(el->type);
        else elemTypes.push_back(CompleteType(BaseType::UNKNOWN));
    }
    node->type = CompleteType(BaseType::TUPLE, std::move(elemTypes));
    return node;
}

std::any ASTBuilder::visitTuple_dec(GazpreaParser::Tuple_decContext *ctx){
    // Build a CompleteType representing the tuple declaration's element types.
    std::vector<CompleteType> elemTypes;
    for (auto typeCtx : ctx->type()){
        auto anyType = visit(typeCtx);
        if (anyType.has_value() && anyType.type() == typeid(CompleteType)){
            elemTypes.push_back(std::any_cast<CompleteType>(anyType));
        } else {
            elemTypes.push_back(CompleteType(BaseType::UNKNOWN));
        }
    }
    return CompleteType(BaseType::TUPLE, std::move(elemTypes));
}






// std::any ASTBuilder::visitStat(GazpreaParser::StatContext *ctx) {
//     // Route to the actual statement inside 'stat: <rule> END'
//     if (ctx->intDec())       return visitIntDec(ctx->intDec());
//     if (ctx->vectorDec())    return visitVectorDec(ctx->vectorDec());
//     if (ctx->assign())       return visitAssign(ctx->assign());
//     if (ctx->cond())         return visitCond(ctx->cond());
//     if (ctx->loop())         return visitLoop(ctx->loop());
//     if (ctx->print())        return visitPrint(ctx->print());
//     return std::shared_ptr<ASTNode>{};
// }

// std::any ASTBuilder::visitExpr(GazpreaParser::ExprContext *ctx) {
//     return visit(ctx->equalityExpr());
// }

// std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext *ctx) {
//     std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
//     auto exprAny = visit(ctx->expr());
//     auto expr = std::any_cast<std::shared_ptr<ExprNode>>(exprAny);
//     auto node = std::make_shared<AssignNode>(id, expr);
//     return std::static_pointer_cast<ASTNode>(node);
// }

// std::any ASTBuilder::visitIntDec(GazpreaParser::IntDecContext *ctx) {
//     std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
//     auto exprAny = visit(ctx->expr());
//     auto expr = std::any_cast<std::shared_ptr<ExprNode>>(exprAny);
//     auto node = std::make_shared<IntDecNode>(id, expr);
//     return std::static_pointer_cast<ASTNode>(node);
// }

// std::any ASTBuilder::visitVectorDec(GazpreaParser::VectorDecContext *ctx) {
//     std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
//     auto exprAny = visit(ctx->expr());
//     auto expr = std::any_cast<std::shared_ptr<ExprNode>>(exprAny);
//     auto node = std::make_shared<VectorDecNode>(id, expr);
//     return std::static_pointer_cast<ASTNode>(node);
// }

// std::any ASTBuilder::visitPrint(GazpreaParser::PrintContext *ctx) {
//     auto exprAny = visit(ctx->expr());
//     auto expr = std::any_cast<std::shared_ptr<ExprNode>>(exprAny);
//     auto node = std::make_shared<PrintNode>(expr);
//     return std::static_pointer_cast<ASTNode>(node);
// }

// std::any ASTBuilder::visitCond(GazpreaParser::CondContext *ctx) {
//     auto condAny = visit(ctx->expr());
//     auto condExpr = std::any_cast<std::shared_ptr<ExprNode>>(condAny);
//     std::vector<std::shared_ptr<ASTNode>> body;
//     for (auto statCtx : ctx->blockStat()) {
//         std::shared_ptr<ASTNode> stmt = std::any_cast<std::shared_ptr<ASTNode>>(visitBlockStat(statCtx));
//         if (stmt) {
//             body.push_back(stmt);
//         }
//     }
//     auto node = std::make_shared<CondNode>(condExpr, std::move(body));
//     return std::static_pointer_cast<ASTNode>(node);
// }

// std::any ASTBuilder::visitLoop(GazpreaParser::LoopContext *ctx) {
//     auto condAny = visit(ctx->expr());
//     auto condExpr = std::any_cast<std::shared_ptr<ExprNode>>(condAny);
//     std::vector<std::shared_ptr<ASTNode>> body;
//     for (auto statCtx : ctx->blockStat()) {
//         std::shared_ptr<ASTNode> stmt = std::any_cast<std::shared_ptr<ASTNode>>(visitBlockStat(statCtx));
//         if (stmt) {
//             body.push_back(std::move(stmt));
//         }
//     }
//     auto node = std::make_shared<LoopNode>(condExpr, std::move(body));
//     return std::static_pointer_cast<ASTNode>(node);
// }

// std::any ASTBuilder::visitBlockStat(GazpreaParser::BlockStatContext *ctx) {
//     // Route to the actual statement inside 'blockStat: <rule> END'
//     if (ctx->assign())   return visitAssign(ctx->assign());
//     if (ctx->cond())     return visitCond(ctx->cond());
//     if (ctx->loop())     return visitLoop(ctx->loop());
//     if (ctx->print())    return visitPrint(ctx->print());
//     return std::shared_ptr<ASTNode>{};
// }

// std::any ASTBuilder::visitEqualityExpr(GazpreaParser::EqualityExprContext *ctx) {
//     size_t n = ctx->comparisonExpr().size();
//     std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitComparisonExpr(ctx->comparisonExpr(0)));
//     for (size_t i = 1; i < n; ++i) {
//         // Find which operator was used at this position (i-1)
//         std::string op;
//         if (ctx->EQEQ(i-1)) {
//             op = ctx->EQEQ(i-1)->getText();
//         } else if (ctx->NEQ(i-1)) {
//             op = ctx->NEQ(i-1)->getText();
//         }
//         auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitComparisonExpr(ctx->comparisonExpr(i)));
//         node = std::make_shared<BinaryOpNode>(node, right, op);
//     }
//     return node;
// }

// std::any ASTBuilder::visitComparisonExpr(GazpreaParser::ComparisonExprContext *ctx) {
//     size_t n = ctx->addSubExpr().size();
//     std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitAddSubExpr(ctx->addSubExpr(0)));
//     for (size_t i = 1; i < n; ++i) {
//         std::string op;
//         if (ctx->LT(i-1)) {
//             op = ctx->LT(i-1)->getText();
//         } else if (ctx->GT(i-1)) {
//             op = ctx->GT(i-1)->getText();
//         }
//         auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitAddSubExpr(ctx->addSubExpr(i)));
//         node = std::make_shared<BinaryOpNode>(node, right, op);
//     }
//     return node;
// }

// std::any ASTBuilder::visitAddSubExpr(GazpreaParser::AddSubExprContext *ctx) {
//     size_t n = ctx->mulDivExpr().size();
//     std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitMulDivExpr(ctx->mulDivExpr(0)));
//     for (size_t i = 1; i < n; ++i) {
//         std::string op;
//         if (ctx->ADD(i-1)) {
//             op = ctx->ADD(i-1)->getText();
//         } else if (ctx->MINUS(i-1)) {
//             op = ctx->MINUS(i-1)->getText();
//         }
//         auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitMulDivExpr(ctx->mulDivExpr(i)));
//         node = std::make_shared<BinaryOpNode>(node, right, op);
//     }
//     return node;
// }

// std::any ASTBuilder::visitMulDivExpr(GazpreaParser::MulDivExprContext *ctx) {
//     size_t n = ctx->rangeExpr().size();
//     std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitRangeExpr(ctx->rangeExpr(0)));
//     for (size_t i = 1; i < n; ++i) {
//         std::string op;
//         if (ctx->MULT(i-1)) {
//             op = ctx->MULT(i-1)->getText();
//         } else if (ctx->DIV(i-1)) {
//             op = ctx->DIV(i-1)->getText();
//         }
//         auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitRangeExpr(ctx->rangeExpr(i)));
//         node = std::make_shared<BinaryOpNode>(node, right, op);
//     }
//     return node;
// }

// std::any ASTBuilder::visitRangeExpr(GazpreaParser::RangeExprContext *ctx) {
//     if (ctx->indexExpr().size() == 1) {
//         // If there's only one indexExpr and no '..', return it directly (do not wrap in RangeNode)
//         return visitIndexExpr(ctx->indexExpr(0));
//     }
//     auto startAny = visitIndexExpr(ctx->indexExpr(0));
//     auto endAny = visitIndexExpr(ctx->indexExpr(1));
//     auto node = std::make_shared<RangeNode>(
//         std::any_cast<std::shared_ptr<ExprNode>>(startAny),
//         std::any_cast<std::shared_ptr<ExprNode>>(endAny)
//     );
//     return std::static_pointer_cast<ExprNode>(node);
// }

// std::any ASTBuilder::visitIndexExpr(GazpreaParser::IndexExprContext *ctx) {
//     // no expr, go to atom
//     if (!ctx->expr()) {
//         return visit(ctx->atom());
//     }

//     auto arrayAny = visit(ctx->atom());
//     auto array = std::any_cast<std::shared_ptr<ExprNode>>(arrayAny);
//     auto index = std::any_cast<std::shared_ptr<ExprNode>>(visit(ctx->expr()));
//     auto node = std::make_shared<IndexNode>(array, index);
//     return std::static_pointer_cast<ExprNode>(node);
// }

// std::any ASTBuilder::visitGenerator(GazpreaParser::GeneratorContext *ctx) {
//     std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
//     auto domAny = visit(ctx->expr(0));
//     auto bodyAny = visit(ctx->expr(1));
//     auto node = std::make_shared<GeneratorNode>(
//         id,
//         std::any_cast<std::shared_ptr<ExprNode>>(domAny),
//         std::any_cast<std::shared_ptr<ExprNode>>(bodyAny));
//     return std::static_pointer_cast<ExprNode>(node);
// }

// std::any ASTBuilder::visitFilter(GazpreaParser::FilterContext *ctx) {
//     std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
//     auto domAny = visit(ctx->expr(0));
//     auto predAny = visit(ctx->expr(1));
//     auto node = std::make_shared<FilterNode>(
//         id,
//         std::any_cast<std::shared_ptr<ExprNode>>(domAny),
//         std::any_cast<std::shared_ptr<ExprNode>>(predAny));
//     return std::static_pointer_cast<ExprNode>(node);
// }

// std::any ASTBuilder::visitAtom(GazpreaParser::AtomContext *ctx) {
//     if (ctx->INT()) {
//         const std::string literal = ctx->INT()->getText();
//         try {
//             size_t parsedChars = 0;
//             int value = std::stoi(literal, &parsedChars, 10);
//             if (parsedChars != literal.size()) {
//                 throw std::runtime_error("TypeError: Integer literal '" + literal + "' is not a valid signed int.");
//             }
//             auto node = std::make_shared<IntNode>(value);
//             return std::static_pointer_cast<ExprNode>(node);
//         } catch (const std::invalid_argument&) {
//             throw std::runtime_error("TypeError: Integer literal '" + literal + "' is not a valid signed int.");
//         } catch (const std::out_of_range&) {
//             throw std::runtime_error("RangeError: Integer literal '" + literal + "' is out of range for signed int.");
//         }
//     } else if (ctx->ID()) {
//         std::string name = ctx->ID()->getText();
//         auto node = std::make_shared<IdNode>(name);
//         return std::static_pointer_cast<ExprNode>(node);
//     } else if (ctx->generator()) {
//         return visitGenerator(ctx->generator());
//     } else if (ctx->filter()) {
//         return visitFilter(ctx->filter());
//     } else if (ctx->expr()) {
//         return visit(ctx->expr());
//     }
//     return std::shared_ptr<ExprNode>{};
// }
