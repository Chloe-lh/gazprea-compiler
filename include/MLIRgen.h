#pragma once

#include "AST.h"
#include "ASTVisitor.h"


#include <string>
#include <unordered_map>
#include <vector>

#include <mlir/IR/Value.h>

// MLIRGen traverses the AST and emits MLIR operations
class MLIRGen : public ASTVisitor {
public:
    explicit MLIRGen(BackEnd& backend);

        // Functions
        void visit(FuncStatNode* node) override; 
        void visit(FuncPrototypeNode* node) override;
        void visit(FuncBlockNode* node) override;
        void visit(ProcedureNode* node) override;
        
        // Declarations
        void visit(TypedDecNode* node) override;
        void visit(InferredDecNode* node) override;
        void visit(TupleTypedDecNode* node) override;
        void visit(TypeAliasDecNode* node) override;
        void visit(TypeAliasNode* node) override;
        void visit(TupleTypeAliasNode* node) override;

        // Statements
        void visit(AssignStatNode* node)    override;
        void visit(OutputStatNode* node)    override;
        void visit(InputStatNode* node)     override;
        void visit(BreakStatNode* node)     override;
        void visit(ContinueStatNode* node)  override;
        void visit(ReturnStatNode* node)    override;
        void visit(CallStatNode* node)      override; 
        void visit(IfNode* node)            override;
        void visit(LoopNode* node)          override;
        void visit(BlockNode* node)         override;


        // Expressions / Operators
        void visit(ParenExpr* node) override;
        void visit(UnaryExpr* node) override;   // unary+, unary-, not
        void visit(ExpExpr* node) override;     // ^
        void visit(MultExpr* node) override;    // *,/,%
        void visit(AddExpr* node) override;     // +, -
        void visit(CompExpr* node) override;    // <, >, <=, >=
        void visit(NotExpr* node) override;     // not
        void visit(EqExpr* node) override;      // ==, !=
        void visit(AndExpr* node) override;     // and
        void visit(OrExpr* node) override;      // or, xor
        void visit(TrueNode* node) override;
        void visit(FalseNode* node) override;
        void visit(CharNode* node) override;
        void visit(IntNode* node) override;
        void visit(IdNode* node) override;
        void visit(TupleLiteralNode* node) override;
        void visit(TupleAccessNode* node) override;
        void visit(TypeCastNode* node) override;
        void visit(TupleTypeCastNode* node) override;
        void visit(RealNode* node) override;

private:
    mlir::Value popValue();
    void pushValue(mlir::Value value);

    BackEnd& backend_;
    mlir::OpBuilder& builder_;
    mlir::ModuleOp module_;
    mlir::MLIRContext& context_;
    mlir::Location loc_;

    // Stack for intermediate MLIR values
    std::vector<mlir::Value> v_stack_;

    // Symbol table for variable values
    std::unordered_map<std::string, mlir::Value> symbolTable_;

    // Memory-backed variables (for cross-region visibility)
    std::unordered_map<std::string, mlir::Value> varMem_; // memref<1xi32> for ints
};
