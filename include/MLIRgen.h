#pragma once

#include "AST.h"
#include "ASTVisitor.h"
#include "Scope.h"
#include "BackEnd.h"


#include <string>
#include <unordered_map>
#include <vector>

// MLIR types. BackEnd.h already includes the necessary MLIR headers; keep
// this include here to satisfy translation units that include MLIRgen.h
// directly.
#include "mlir/IR/Value.h"

// MLIRGen traverses the AST and emits MLIR operations
class MLIRGen : public ASTVisitor {
public:
    explicit MLIRGen(BackEnd& backend);
    explicit MLIRGen(BackEnd& backend, Scope* rootScope, const std::unordered_map<const ASTNode*, Scope*>* scopeMap);

    void visit(FileNode* node) override;

    // Functions
    void visit(FuncStatNode* node) override;
    void visit(FuncPrototypeNode* node) override;
    void visit(FuncBlockNode* node) override;
    void visit(ProcedureBlockNode* node) override;
    void visit(ProcedurePrototypeNode* node) override;

    // Declarations
    void visit(TypedDecNode* node) override;
    void visit(InferredDecNode* node) override;
    void visit(TupleTypedDecNode* node) override;
    void visit(StructTypedDecNode* node) override;
    void visit(TypeAliasDecNode* node) override;
    void visit(TypeAliasNode* node) override;
    void visit(TupleTypeAliasNode* node) override;

    // Statements
    void visit(AssignStatNode* node)    override;
    void visit(DestructAssignStatNode* node) override;
    void visit(TupleAccessAssignStatNode* node) override;
    void visit(StructAccessAssignStatNode* node) override;
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
    void visit(FuncCallExprOrStructLiteral* node) override;
    void visit(UnaryExpr* node) override;   // unary+, unary-, not
    void visit(ExpExpr* node) override;     // ^
    void visit(MultExpr* node) override;    // *,/,%
    void visit(AddExpr* node) override;     // +, -
    void visit(CompExpr* node) override;    // <, >, <=, >=
    void visit(NotExpr* node) override;     // not
    void visit(EqExpr* node) override;      // ==, !=
    void visit(AndExpr* node) override;     // and
    void visit(OrExpr* node) override;      // or, xor
    void visit (StructAccessNode* node) override;
    void visit(TupleAccessNode* node) override;
    void visit(TypeCastNode* node) override;
    void visit(TupleTypeCastNode* node) override;

    // variables
    void visit(IdNode* node) override;

    // Primitives
    void visit(TrueNode* node) override;
    void visit(FalseNode* node) override;
    void visit(CharNode* node) override;
    void visit(IntNode* node) override;
    void visit(RealNode* node) override;
    void visit(StringNode* node) override;
    void visit(TupleLiteralNode* node) override;

    // arrays
    void visit(ArrayStrideExpr *node) override;
    void visit(ArraySliceExpr *node) override;
    void visit(ArrayAccessExpr *node) override;
    void visit(ArrayTypedDecNode *node) override;
    void visit(ArrayTypeNode *node) override;
    void visit(ExprListNode *node) override;
    void visit(ArrayLiteralNode *node) override;
    void visit(RangeExprNode *node) override;

    // helpers
    void assignTo(VarInfo* literal, VarInfo* variable, int line);
    void allocaLiteral(VarInfo* varInfo, int line);
    void allocaVar(VarInfo* varInfo, int line);
    void zeroInitializeVar(VarInfo* varInfo);
    VarInfo castType(VarInfo* from, CompleteType* to, int line);
    VarInfo promoteType(VarInfo* from, CompleteType* to, int line);

    // Globals helpers
    mlir::Type getLLVMType(const CompleteType& type);
    mlir::Attribute extractConstantValue(std::shared_ptr<ExprNode> expr, const CompleteType& targetType);
    mlir::Value createGlobalVariable(const std::string& name, const CompleteType& type, bool isConst, mlir::Attribute initValue = nullptr);

    // Helpers to honour compile-time constants produced by ConstantFolding.
    // If the constant can be represented as a literal memref, create it and
    // return a VarInfo ready to be pushed on the value stack.
    // Throws on unsupported types; callers should handle errors.
    VarInfo createLiteralFromConstant(const ConstantValue &cv, const CompleteType &type, int line);
    // if `node` has a compile-time constant, emit the
    // corresponding literal and push it on the value stack; returns true on
    // success and false if no constant or unsupported constant type.
    bool tryEmitConstantForNode(ExprNode* node);
    mlir::func::FuncOp createFunctionDeclaration(const std::string &name,
                                                 const std::vector<VarInfo> &params,
                                                 const CompleteType &returnType);
    mlir::func::FuncOp beginFunctionDefinition(
        const ASTNode* funcOrProc,
        const std::string &name,
        const std::vector<VarInfo> &params,
        const CompleteType &returnType,
        Scope*& savedScope
    );
    // Bind parameters (store entry block args into each VarInfo's memref storage)
    void bindFunctionParameters(mlir::func::FuncOp func, const std::vector<VarInfo> &params);
    // use constant folding
    void bindFunctionParametersWithConstants(mlir::func::FuncOp func, const std::vector<VarInfo> &params, int line);
    void lowerFunctionOrProcedureBody(const std::vector<VarInfo> &params, std::shared_ptr<BlockNode> body, const CompleteType &returnType, Scope* savedScope) ;
    mlir::func::FuncOp beginFunctionDefinitionWithConstants(
        const ASTNode* funcOrProc,
        const std::string &name,
        const std::vector<VarInfo> &params,
        const CompleteType &returnType,
        Scope* &savedScope);
    mlir::Value getSSAValue(const VarInfo &v);


private:
    VarInfo popValue();
    void pushValue(VarInfo& value);

    BackEnd& backend_;
    mlir::OpBuilder& builder_;
    mlir::OpBuilder allocaBuilder_;  // only for allocas
    mlir::ModuleOp module_;
    mlir::MLIRContext& context_;
    mlir::Location loc_;

    // Stack for intermediate MLIR values
    std::vector<VarInfo> v_stack_;

    // Storing named values + types
    Scope* root_;
    Scope* currScope_;
    const std::unordered_map<const ASTNode*, Scope*>* scopeMap_;

    // Loop control flow tracking
    struct LoopContext {
        mlir::Block* exitBlock;      // Block after the loop (for break)
        mlir::Block* continueBlock;  // Block to branch to for continue
    };
    std::vector<LoopContext> loopContexts_;

};
