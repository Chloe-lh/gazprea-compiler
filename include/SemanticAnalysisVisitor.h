#include "ASTVisitor.h"
#include "AST.h"
#include "Scope.h"
#include <unordered_map>


class SemanticAnalysisVisitor: public ASTVisitor {
    public:
        void visit(FileNode* node) override;

        // Functions
        virtual void visit(FuncStatNode* node) override; 
        virtual void visit(FuncPrototypeNode* node) override;
        virtual void visit(FuncBlockNode* node) override;
        virtual void visit(ProcedureNode* node) override;
        
        // Declarations
        virtual void visit(TypedDecNode* node) override;
        virtual void visit(InferredDecNode* node) override;
        virtual void visit(TupleTypedDecNode* node) override;
        virtual void visit(TypeAliasDecNode* node) override;
        virtual void visit(TypeAliasNode* node) override;
        virtual void visit(TupleTypeAliasNode* node) override;

        // Statements
        virtual void visit(AssignStatNode* node)    override;
        virtual void visit(OutputStatNode* node)    override;
        virtual void visit(InputStatNode* node)     override;
        virtual void visit(BreakStatNode* node)     override;
        virtual void visit(ContinueStatNode* node)  override;
        virtual void visit(ReturnStatNode* node)    override;
        virtual void visit(CallStatNode* node)      override; // TODO: not allowed within functions. 
        virtual void visit(IfNode* node)            override;
        virtual void visit(LoopNode* node)          override;
        virtual void visit(BlockNode* node)         override;


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
        void visit(FuncCallExpr* node) override;
        void visit(RealNode* node) override;



    private:
        // Persistent scope tree and context index
        std::unique_ptr<Scope> root_;
        Scope* current_ = nullptr;
        std::unordered_map<const ASTNode*, Scope*> scopeByCtx_;
        bool seenMain_ = false;

        void throwOperandError(const std::string op, const std::vector<CompleteType>& operands, std::string additionalInfo);
        void handleAssignError(const std::string varName, const CompleteType &varType, const CompleteType &exprType);

        // Helpers
        void enterScopeFor(const ASTNode* ownerCtx, const bool inLoop, const CompleteType* returnType);
        void exitScope();
        bool guaranteesReturn(const BlockNode* block) const;
};
