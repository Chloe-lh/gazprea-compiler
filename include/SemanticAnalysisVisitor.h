#include "ASTVisitor.h"
#include "AST.h"
#include "Scope.h"
#include <unordered_map>
#include <memory>

// Helper for resolving alias types that were marked BaseType::UNRESOLVED during building
CompleteType resolveUnresolvedType(Scope *scope, const CompleteType &t);

class SemanticAnalysisVisitor: public ASTVisitor {
    public:
        Scope* getRootScope();
        const std::unordered_map<const ASTNode*, Scope*>& getScopeMap() const;

        void visit(FileNode* node) override;

        // Functions
        virtual void visit(FuncStatNode* node) override; 
        virtual void visit(FuncPrototypeNode* node) override;
        virtual void visit(FuncBlockNode* node) override;
        virtual void visit(ProcedureBlockNode* node) override;
        virtual void visit(ProcedurePrototypeNode* node) override;
        virtual void visit(BuiltInFuncNode* node) override;

        
        // Declarations
        virtual void visit(TypedDecNode* node) override;
        virtual void visit(InferredDecNode* node) override;
        virtual void visit(TupleTypedDecNode* node) override;
        virtual void visit(StructTypedDecNode *node) override;
        virtual void visit(TypeAliasDecNode* node) override;
        virtual void visit(TypeAliasNode* node) override;
        virtual void visit(TupleTypeAliasNode* node) override;

        // Statements
        virtual void visit(AssignStatNode* node)    override;
        virtual void visit(DestructAssignStatNode* node) override;
        virtual void visit(TupleAccessAssignStatNode* node) override;
        virtual void visit(StructAccessAssignStatNode* node) override;
        virtual void visit(ArrayAccessAssignStatNode* node) override;
        virtual void visit(OutputStatNode* node)    override;
        virtual void visit(InputStatNode* node)     override;
        virtual void visit(BreakStatNode* node)     override;
        virtual void visit(ContinueStatNode* node)  override;
        virtual void visit(ReturnStatNode* node)    override;
        virtual void visit(CallStatNode* node)      override;
        virtual void visit(IfNode* node)            override;
        virtual void visit(LoopNode* node)          override;
        virtual void visit(BlockNode* node)         override;


        // Expressions / Operators
        void visit(ParenExpr* node) override;
        void visit(UnaryExpr* node) override;   // unary+, unary-, not
        void visit(DotExpr* node) override;
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
        void visit(StructAccessNode *node) override;
        void visit(TypeCastNode* node) override;
        void visit(TupleTypeCastNode* node) override;
        void visit(FuncCallExprOrStructLiteral* node) override;
        void visit(RealNode* node) override;
        void visit(StringNode* node) override;

        //arrays
        void visit(ArrayStrideExpr *node) override;
        void visit(ArraySliceExpr *node) override;
        void visit(ArrayAccessNode *node) override;
        void visit(ArrayTypedDecNode *node) override;
        void visit(ExprListNode *node) override;
        void visit(ArrayLiteralNode *node) override;
        void visit(RangeExprNode *node) override;


    private:
        // Persistent scope tree and context index
        std::unique_ptr<Scope> root_;
        Scope* current_ = nullptr;
        std::unordered_map<const ASTNode*, Scope*> scopeByCtx_;
        bool seenMain_ = false;

        void throwOperandError(const std::string op, const std::vector<CompleteType>& operands, std::string additionalInfo, int line);
        void handleAssignError(const std::string varName, const CompleteType &varType, const CompleteType &exprType, int line);

        // Helpers
        void enterScopeFor(const ASTNode* ownerCtx, const bool inLoop, const CompleteType* returnType);
        void exitScope();
        bool guaranteesReturn(const BlockNode* block) const;
        void handleGlobalErrors(DecNode *node);
        CompleteType resolveUnresolvedType(Scope *scope, const CompleteType &t, int line);

};
