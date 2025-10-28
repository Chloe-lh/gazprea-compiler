#pragma once
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "Types.h"

//abstract class that is extended by the different passes in the pipeline
class ASTVisitor;
//forward declarations
class TupleDecNode;

class ASTNode{ //virtual class
    public:
        // type
        virtual ~ASTNode() = default;
        virtual void accept(ASTVisitor& visitor) = 0;
        ValueType type = ValueType::UNKNOWN;
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
// statements
class StatNode : public ASTNode {
public:
    virtual ~StatNode() = default;
    virtual void accept(ASTVisitor& visitor) = 0;
};
// literal expressions
class LiteralExprNode : public ExprNode {
public:
    virtual ~LiteralExprNode() = default;
};
// expressions
class UnaryExprNode : public ExprNode {
public:
    std::string op;
    std::unique_ptr<ExprNode> operand;
    UnaryExprNode(const std::string& op, std::unique_ptr<ExprNode> operand)
        : op(op), operand(std::move(operand)) {}
};
//  binary expressions
class BinaryExprNode : public ExprNode {
public:
    std::string op;
    std::unique_ptr<ExprNode> left;
    std::unique_ptr<ExprNode> right;
    BinaryExprNode(const std::string& op, std::unique_ptr<ExprNode> left, std::unique_ptr<ExprNode> right)
        : op(op), left(std::move(left)), right(std::move(right)) {}
};
// functions
class FuncNode : public ASTNode {
public:
    std::string name;
    std::vector<std::pair<std::string, std::string>> parameters; // (type, name)
    std::string returnType;
    std::unique_ptr<TupleDecNode> returnTupleType; // optional
    std::unique_ptr<BlockNode> body; // optional
    std::unique_ptr<StatNode> returnStat; // optional
    bool isPrototype = false;
    bool isTupleReturn = false;
};
// function classes

// expression classes
class ParenExpr: public ExprNode {
    public: 
        explicit ParenExpr();
        void accept(ASTVisitor& visitor) override;
};
class UnaryExpr: public UnaryExprNode {
    public:
        UnaryExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor) override;
};
class ExpExpr: public BinaryExprNode {
    public:
        ExpExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor) override;
};
class MultExpr: public BinaryExprNode {
    public:
        MultExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor)override;
};
class AddExpr: public BinaryExprNode {
    public:
        AddExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor) override;
};
class CompExpr: public BinaryExprNode {
    public:
        CompExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor)override;
};
class NotExpr: public UnaryExprNode {
    public:
        NotExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor)override;
};
class EqExpr: public BinaryExprNode {
    public:
        EqExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor)override;
};
class AndExpr: public BinaryExprNode {
    public:
        AndExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor)override;
};
class OrExpr: public BinaryExprNode {
    public:
        OrExpr(const std::string& op, std::unique_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor)override;
};
class TrueNode: public LiteralExprNode {
    public:
        bool value;
        explicit TrueNode() : value(true) {}
        void accept(ASTVisitor& visitor)override;
};
class FalseNode: public LiteralExprNode {
    public:
        bool value;
        explicit FalseNode() : value(false) {}
        void accept(ASTVisitor& visitor)override;
};
class CharNode: public LiteralExprNode {
    public:
        char value;
        explicit CharNode(char v);
        void accept(ASTVisitor& visitor)override;
};
class IntNode: public LiteralExprNode {
    public:
        int value;
        explicit IntNode(int v); 
        void accept(ASTVisitor& visitor) override;
};
class IdNode: public ExprNode {
    public:
        const std::string id;
        explicit IdNode(const std::string& id);
        void accept(ASTVisitor& visitor) override;
};
//declaration classes
class TupleDecNode : public DecNode {
    public:
        std::vector<std::string> elementTypes;
        TupleDecNode(const std::string& name, std::vector<std::string>& types);
        void accept(ASTVisitor& visitor) override;
};
class TypedDecNode : public DecNode {
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
class InferredDecNode : public DecNode {
    public:
        std::string qualifier;
        std::unique_ptr<ExprNode> init;
        InferredDecNode(const std::string& name, const std::string& qualifier, std::unique_ptr<ExprNode> init);
        void accept(ASTVisitor& visitor) override;
};
// statement nodes
class AssignStatNode: public ASTNode{
    public:
        std::string name;
        std::unique_ptr<ExprNode> expr;
        AssignStatNode(const std::string& name, std::unique_ptr<ExprNode> expr);
        void accept(ASTVisitor& visitor) override;
};
class OutputStatNode: public ASTNode{
    public:
        std::unique_ptr<ExprNode> expr;
        OutputStatNode(std::unique_ptr<ExprNode> expr);
        void accept(ASTVisitor& visitor) override;
};
class InputStatNode: public ASTNode{
    public:
        std::string name;
        explicit InputStatNode(const std::string& name) value(name);
};
class BreakStatNode: public ASTNode{
    public:
        BreakStatNode() = default;
        void accept(ASTVisitor& visitor) override;
};
class ContinueStatNode: public ASTNode{
    public:
        ContinueStatNode() = default;
        void accept(ASTVisitor& visitor) override;
};
class ReturnStatNode: public ASTNode{
    public:
        ReturnStatNode() = default;
        void accept(ASTVisitor& visitor) override;
};
class CallStatNode{
    public:
        std::string funcName;
        std::vector<std::unique_ptr<ExprNode>> args;
        CallStatNode(const std::string& funcName, std::vector<std::unique_ptr<ExprNode>> args)
        void accept(ASTVisitor& visitor) override;
};

class FileNode : public ASTNode {
    public:
        explicit FileNode();
        void accept(ASTVisitor& visitor) override;
};






