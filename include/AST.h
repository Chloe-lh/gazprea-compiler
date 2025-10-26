#pragma once
#include <string>
#include <vector>
#include <memory>
#include <utility>

//abstract class that is extended by the different passes in the pipeline
class ASTVisitor;

class ASTNode{ //virtual class
    public:
        // type
        virtual ~ASTNode() = default;
        virtual void accept(ASTVisitor& visitor) = 0;
};
// super classes
class DecNode: public ASTNode{
    public:
        std::string name;
        std::string type;
        virtual ~DecNode() = default;
        virtual void accept(ASTVisitor& visitor) = 0;
        // SCOPE
};
class ExprNode: public ASTNode {
    public:
        virtual ~ExprNode() = default;
        virtual void accept(ASTVisitor& visitor) = 0;
};
class UnaryExprNode: public ExprNode{
    public:
        std::string op;
};
class BinaryExprNode:public ExprNode{
    public:
        std::string op;
        std::unique_ptr<ExprNode> left;
        std::unique_ptr<ExprNode> right;
};

// expression classes
class ParenExpr: public ExprNode{
    public: 
        explicit ParenExpr();
        void accept(ASTVisitor& visitor) override;
};
class UnaryExpr: public UnaryExprNode{
    public:
        UnaryExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor) override;
};
class ExpExpr: public BinaryExprNode{
    public:
        ExpExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor) override;
};
class MultExpr: public BinaryExprNode{
    public:
        MultExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor)override;
};
class AddExpr: public BinaryExprNode{
    public:
        AddExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor) override;
};
class CompExpr: public BinaryExprNode{
    public:
        CompExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor)override;
};
class NotExpr: public UnaryExprNode{
    public:
        NotExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor)override;
};
class EqExpr: public BinaryExprNode{
    public:
        EqExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor)override;
};
class AndExpr: public BinaryExprNode{
    public:
        AndExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor)override;
};
class OrExpr: public BinaryExprNode{
    public:
        OrExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor)override;
};
class TrueNode: public ExprNode{
    public:
        bool value;
        explicit TrueNode() : value(true) {}
        void accept(ASTVisitor& visitor)override;
};
class FalseNode: public ExprNode{
    public:
        bool value;
        explicit FalseNode() : value(false) {}
        void accept(ASTVisitor& visitor)override;
};
class CharNode: public ExprNode{
    public:
        char value;
        explicit CharNode(char v);
        void accept(ASTVisitor& visitor)override;
};
class IntNode : public ExprNode {
    public:
        int value;
        explicit IntNode(int v); 
        void accept(ASTVisitor& visitor) override;
};
class IdNode : public ExprNode {
    public:
        const std::string id;
        explicit IdNode(const std::string& id);
        void accept(ASTVisitor& visitor) override;
};
//declaration classes
class TupleDecNode: public DecNode{
    public:
        std::vector<std::string> elementTypes;
        TupleDecNode(const std::string& name, std::vector<std::string>& types);
        void accept(ASTVisitor& visitor) override;
};
class TypedDecNode: public DecNode{
public:
    std::string qualifier; // optional
    std::string type_alias; // type OR alias
    std::unique_ptr<ExprNode> init; // optional initializer
    TypedDecNode(
        const std::string& name,
        const std::string& type_alias,
        std::unique_ptr<ExprNode> init = nullptr,
        const std::string& qualifier = ""
    );
    void accept(ASTVisitor& visitor) override;
};
class InferredDecNode: public DecNode{
    public:
        std::string qualifier;
        std::unique_ptr<ExprNode> init;
        InferredDecNode(const std::string& name, const std::string& qualifier, std::unique_ptr<ExprNode> init);
        void accept(ASTVisitor& visitor) override;
};

class FileNode: public ASTNode{
    public:
        explicit FileNode();
        void accept(ASTVisitor& visitor) override;
};






