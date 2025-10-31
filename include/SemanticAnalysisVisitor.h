#include "ASTVisitor.h"
#include "AST.h"
#include "Scope.h"
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <set>
#include <iterator>

class SemanticAnalysisVisitor: public ASTVisitor {
    public:
        void visit(FileNode* node) override;

        // Functions
        virtual void visit(FuncStatNode* node); // TODO: ensure has return in all paths
        virtual void visit(FuncPrototypeNode* node);
        virtual void visit(FuncBlockNode* node);
        virtual void visit(ProcedureNode* node);
        
        // Declarations
        virtual void visit(TypedDecNode* node) override;
        virtual void visit(InferredDecNode* node) override;
        virtual void visit(TupleTypedDecNode* node) override;
        virtual void visit(TypeAliasDecNode* node) override;
        virtual void visit(TypeAliasNode* node);
        virtual void visit(TupleTypeAliasNode* node) override;

        // Statements
        virtual void visit(AssignStatNode* node);
        virtual void visit(OutputStatNode* node);
        virtual void visit(InputStatNode* node);
        virtual void visit(BreakStatNode* node);
        virtual void visit(ContinueStatNode* node);
        virtual void visit(ReturnStatNode* node);
        virtual void visit(CallStatNode* node);
        virtual void visit(IfNode* node) = 0;
        virtual void visit(LoopNode* node) = 0;
        virtual void visit(BlockNode* node);


        // Operators
        void visit(UnaryExpr* node) override;   // unary+, unary-, not
        void visit(ExpExpr* node) override;     // ^
        void visit(MultExpr* node) override;    // *,/,%
        void visit(AddExpr* node) override;     // +, -
        void visit(CompExpr* node) override;    // <, >, <=, >=
        void visit(EqExpr* node) override;      // ==, !=
        void visit(AndExpr* node) override;     // and
        void visit(OrExpr* node) override;      // or, xor



    private:
        // Persistent scope tree and context index
        std::unique_ptr<Scope> root_;
        Scope* current_ = nullptr;
        std::unordered_map<const ASTNode*, Scope*> scopeByCtx_;

        void throwOperandError(const std::string op, const std::vector<CompleteType>& operands, std::string additionalInfo);
        void handleAssignError(const std::string varName, const CompleteType &varType, const CompleteType &exprType);

        // Helpers
        void enterScopeFor(const ASTNode* ownerCtx, const bool inLoop, const CompleteType* returnType);
        void exitScope();
        bool guaranteesReturn(const BlockNode* block) const;
};
