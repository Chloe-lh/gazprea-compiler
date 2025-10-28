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
class BlockNode;


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
class FuncBlockNode : public FuncNode {
    public:
        FuncBlockNode(
            const std::string& name,
            const std::vector<std::pair<std::string, std::string>>& parameters,
            const std::string& returnType,
            std::unique_ptr<BlockNode> body
        );
        void accept(ASTVisitor& visitor) override;
};

class FuncBlockTupleReturnNode : public FuncNode {
    public:
        FuncBlockTupleReturnNode(
            const std::string& name,
            const std::vector<std::pair<std::string, std::string>>& parameters,
            std::unique_ptr<TupleDecNode> returnTupleType,
            std::unique_ptr<BlockNode> body
        );
        void accept(ASTVisitor& visitor) override;
};

class FuncStatNode : public FuncNode {
public:
    FuncStatNode(
        const std::string& name,
        const std::vector<std::pair<std::string, std::string>>& parameters,
        const std::string& returnType,
        std::unique_ptr<StatNode> returnStat
    );
    void accept(ASTVisitor& visitor) override;
};

class FuncPrototypeNode : public FuncNode {
public:
    FuncPrototypeNode(
        const std::string& name,
        const std::vector<std::pair<std::string, std::string>>& parameters,
        const std::string& returnType
    );
    void accept(ASTVisitor& visitor) override;
};

class FuncPrototypeTupleReturnNode : public FuncNode {
public:
    FuncPrototypeTupleReturnNode(
        const std::string& name,
        const std::vector<std::pair<std::string, std::string>>& parameters,
        std::unique_ptr<TupleDecNode> returnTupleType
    );
    void accept(ASTVisitor& visitor) override;
};

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
        AndExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor)override;
};
class OrExpr: public BinaryExprNode {
    public:
        OrExpr(const std::string& op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r);
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
        const std::string& qualifier = "",
        std::unique_ptr<ExprNode> init = nullptr
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
class InputStatNode: public StatNode {
public:
    std::string name;
    explicit InputStatNode(const std::string& name) : name(name) {}
    void accept(ASTVisitor& visitor) override;
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
class CallStatNode : public ASTNode {
public:
    std::string funcName;
    std::vector<std::unique_ptr<ExprNode>> args;
    CallStatNode(const std::string& funcName, std::vector<std::unique_ptr<ExprNode>> args);
    void accept(ASTVisitor& visitor) override;
};

class IfNode : public StatNode {
public:
    std::unique_ptr<ExprNode> condition;
    std::unique_ptr<BlockNode> thenBlock;
    std::unique_ptr<BlockNode> elseBlock; // optional
    IfNode(std::unique_ptr<ExprNode> condition,
           std::unique_ptr<BlockNode> thenBlock,
           std::unique_ptr<BlockNode> elseBlock = nullptr)
        : condition(std::move(condition)),
          thenBlock(std::move(thenBlock)),
          elseBlock(std::move(elseBlock)) {}
    void accept(ASTVisitor& visitor) override;
};

class LoopNode : public StatNode {
public:
    std::unique_ptr<BlockNode> body;
    std::unique_ptr<ExprNode> condition; // optional (for while)
    LoopNode(std::unique_ptr<BlockNode> body,
             std::unique_ptr<ExprNode> condition = nullptr)
        : body(std::move(body)), condition(std::move(condition)) {}
    void accept(ASTVisitor& visitor) override;
};

class BlockNode : public ASTNode {
    public:
        std::vector<std::unique_ptr<DecNode>> decs;
        std::vector<std::unique_ptr<StatNode>> stats;

        BlockNode(
            std::vector<std::unique_ptr<DecNode>> declarations,
            std::vector<std::unique_ptr<StatNode>> statements
        )
            : decs(std::move(declarations)), stats(std::move(statements)) {}
        void accept(ASTVisitor& visitor) override;
};
class FileNode : public ASTNode {
    public:
        explicit FileNode();
        void accept(ASTVisitor& visitor) override;
};

class ProcedureNode : public ASTNode {
public:
    std::string name;
    std::vector<std::pair<std::string, std::string>> parameters;
    std::unique_ptr<BlockNode> body;
    ProcedureNode(
        const std::string& name,
        const std::vector<std::pair<std::string, std::string>>& parameters,
        std::unique_ptr<BlockNode> body
    );
    void accept(ASTVisitor& visitor) override;
};

class TypeAliasNode : public ASTNode {
public:
    std::string aliasName;
    std::string typeName;
    TypeAliasNode(const std::string& aliasName, const std::string& typeName);
    void accept(ASTVisitor& visitor) override;
};

class TupleTypeAliasNode : public ASTNode {
public:
    std::string aliasName;
    std::unique_ptr<TupleDecNode> tupleType;
    TupleTypeAliasNode(const std::string& aliasName, std::unique_ptr<TupleDecNode> tupleType);
    void accept(ASTVisitor& visitor) override;
};

class TupleLiteralNode : public ExprNode {
public:
    std::vector<std::unique_ptr<ExprNode>> elements;
    TupleLiteralNode(std::vector<std::unique_ptr<ExprNode>> elements);
    void accept(ASTVisitor& visitor) override;
};

class TupleAccessNode : public ExprNode {
public:
    std::string tupleName;
    int index;
    TupleAccessNode(const std::string& tupleName, int index);
    void accept(ASTVisitor& visitor) override;
};

class TypeCastNode : public ExprNode {
public:
    std::string targetType;
    std::unique_ptr<ExprNode> expr;
    TypeCastNode(const std::string& targetType, std::unique_ptr<ExprNode> expr);
    void accept(ASTVisitor& visitor) override;
};

class TupleTypeCastNode : public ExprNode {
public:
    std::unique_ptr<TupleDecNode> targetTupleType;
    std::unique_ptr<ExprNode> expr;
    TupleTypeCastNode(std::unique_ptr<TupleDecNode> targetTupleType, std::unique_ptr<ExprNode> expr);
    void accept(ASTVisitor& visitor) override;
};

class RealNode : public LiteralExprNode {
public:
    double value;
    explicit RealNode(double value);
    void accept(ASTVisitor& visitor) override;
};







