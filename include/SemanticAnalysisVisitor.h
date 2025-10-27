#include "ASTVisitor.h"
#include "AST.h"
#include "Scope.h"
#include <unordered_map>

class SemanticAnalysisVisitor: ASTVisitor {
    public:
        void visit(FileNode* node) override;
        void visit(CondNode* node) override;
        void visit(LoopNode* node) override;
        void visit(IntDecNode* node) override;
        void visit(VectorDecNode* node) override;
        void visit(AssignNode* node) override;

        // Values
        void visit(IntNode* node) override;
        void visit(IdNode* node) override;

        // expressions 
        void visit(BinaryOpNode* node) override;
        void visit(RangeNode* node) override;
        void visit(IndexNode* node) override;

        // generators
        void visit(GeneratorNode* node) override;
        void visit(FilterNode* node) override; 
        
        void visit(PrintNode* node) override;

        // New gazprea methods
        void visit(UnaryExpr* node) override;   // unary+, unary-, not
        void visit(ExpExpr* node) override;     // ^
        void visit(MultExpr* node) override;    // *,/,%

    private:
        // Persistent scope tree and context index
        std::unique_ptr<Scope> root_;
        Scope* current_ = nullptr;
        std::unordered_map<const ASTNode*, Scope*> scopeByCtx_;

        void throwOperandError(const std::string op, const std::vector<ValueType>& operands, std::string additionalInfo);

        void enterScopeFor(const ASTNode* ownerCtx);
        void exitScope();
};