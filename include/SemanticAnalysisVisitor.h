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

        void enterScopeFor(const ASTNode* ownerCtx);
        void exitScope();
};