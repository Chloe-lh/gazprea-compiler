// Tuple type node: represents a tuple type signature (e.g., tuple(int, char))
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "Types.h"
#include "Scope.h" 

//abstract class that is extended by the different passes in the pipeline
class ASTVisitor;
//forward declarations
class BlockNode;
class TypeAliasNode;


class ASTNode{ //virtual class
    public:
        // type
        virtual ~ASTNode() = default;
        virtual void accept(ASTVisitor& visitor) = 0;
        CompleteType type = CompleteType(BaseType::UNKNOWN);
};
// super classes
class DecNode: public ASTNode{
    public:
        std::string name;
        std::string declTypeName; // Renamed from 'type' to 'declTypeName'
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
    std::shared_ptr<ExprNode> operand;
    UnaryExprNode(const std::string& op, std::shared_ptr<ExprNode> operand)
        : op(op), operand(std::move(operand)) {}
};
//  binary expressions
class BinaryExprNode : public ExprNode {
public:
    std::string op;
    std::shared_ptr<ExprNode> left;
    std::shared_ptr<ExprNode> right;
    BinaryExprNode(const std::string& op, std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right)
        : op(op), left(std::move(left)), right(std::move(right)) {}
};
// functions
class FuncNode : public ASTNode {
public:
    std::string name;
    std::vector<VarInfo> parameters; //  prototypes may omit identifier
    CompleteType returnType; // optional
    std::shared_ptr<BlockNode> body; // optional
    std::shared_ptr<StatNode> returnStat; // optional

    FuncNode(
        const std::string& name,
        const std::vector<VarInfo>& parameters,
        CompleteType returnType,
        std::shared_ptr<BlockNode> body = nullptr,
        std::shared_ptr<StatNode> returnStat = nullptr
    ) : name(name), parameters(parameters), returnType(std::move(returnType)), body(std::move(body)),
        returnStat(std::move(returnStat)) {}
};

class FuncStatNode : public FuncNode {
public:
    FuncStatNode(
        const std::string& name,
        const std::vector<VarInfo>& parameters,
        CompleteType returnType,
        std::shared_ptr<StatNode> returnStat
    );
    void accept(ASTVisitor& visitor) override;
};

class FuncPrototypeNode : public FuncNode {
public:
    FuncPrototypeNode(
        const std::string& name,
        const std::vector<VarInfo>& parameters,
        CompleteType returnType
    );
    void accept(ASTVisitor& visitor) override;
};

// Functions with a block body
class FuncBlockNode : public FuncNode {
public:
    FuncBlockNode(
        const std::string& name,
        const std::vector<VarInfo>& parameters,
        CompleteType returnType,
        std::shared_ptr<BlockNode> body
    );
    void accept(ASTVisitor& visitor) override;
};

// expression classes
class ParenExpr: public ExprNode {
    public: 
        std::shared_ptr<ExprNode> expr;
        explicit ParenExpr(std::shared_ptr<ExprNode> expr);
        void accept(ASTVisitor& visitor) override;
};
class UnaryExpr: public UnaryExprNode {
    public:
        UnaryExpr(const std::string& op, std::shared_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor) override;
};
class ExpExpr: public BinaryExprNode {
    public:
        ExpExpr(const std::string& op, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor) override;
};
class MultExpr: public BinaryExprNode {
    public:
        MultExpr(const std::string& op, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor)override;
};
class AddExpr: public BinaryExprNode {
    public:
        AddExpr(const std::string& op, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor) override;
};
class CompExpr: public BinaryExprNode {
    public:
        CompExpr(const std::string& op, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor)override;
};
class NotExpr: public UnaryExprNode {
    public:
        NotExpr(const std::string& op, std::shared_ptr<ExprNode> operand);
        void accept(ASTVisitor& visitor)override;
};
class EqExpr: public BinaryExprNode {
    public:
        EqExpr(const std::string& op, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor)override;
};
class AndExpr: public BinaryExprNode {
    public:
        AndExpr(const std::string& op, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r);
        void accept(ASTVisitor& visitor)override;
};
class OrExpr: public BinaryExprNode {
    public:
        OrExpr(const std::string& op, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r);
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

class TypeAliasDecNode: public DecNode {
    public: 
        std::string alias;

    TypeAliasDecNode(const std::string& alias, const CompleteType& type);
    void accept(ASTVisitor& visitor) override;
};

class TupleTypedDecNode : public DecNode {
public:
    std::shared_ptr<ExprNode> init;
    TupleTypedDecNode(const std::string& name, CompleteType tupleType);
    void accept(ASTVisitor& visitor) override;
};

class TypedDecNode : public DecNode {
public:
    std::string qualifier; // optional
    std::shared_ptr<TypeAliasNode> type_alias; // type OR alias
    std::shared_ptr<ExprNode> init; // optional initializer
    TypedDecNode(
        const std::string& name,
        std::shared_ptr<TypeAliasNode> type_alias,
        const std::string& qualifier = "",
        std::shared_ptr<ExprNode> init = nullptr
    );
    void accept(ASTVisitor& visitor) override;
};
class InferredDecNode : public DecNode {
    public:
        std::string qualifier;
        std::shared_ptr<ExprNode> init;
        InferredDecNode(const std::string& name, const std::string& qualifier, std::shared_ptr<ExprNode> init);
        void accept(ASTVisitor& visitor) override;
};
// statement nodes
class AssignStatNode: public StatNode{
    public:
        std::string name;
        std::shared_ptr<ExprNode> expr;
        AssignStatNode(const std::string& name, std::shared_ptr<ExprNode> expr);
        void accept(ASTVisitor& visitor) override;
};
class OutputStatNode: public StatNode{
    public:
        std::shared_ptr<ExprNode> expr;
        OutputStatNode(std::shared_ptr<ExprNode> expr);
        void accept(ASTVisitor& visitor) override;
};
class InputStatNode: public StatNode {
    public:
        std::string name;
        explicit InputStatNode(const std::string& name) : name(name) {}
        void accept(ASTVisitor& visitor) override;
};
class BreakStatNode: public StatNode{
    public:
        BreakStatNode() = default;
        void accept(ASTVisitor& visitor) override;
};
class ContinueStatNode: public StatNode{
    public:
        ContinueStatNode() = default;
        void accept(ASTVisitor& visitor) override;
};
class ReturnStatNode: public StatNode{
    public:
        std::shared_ptr<ExprNode> expr;
        ReturnStatNode(std::shared_ptr<ExprNode> expr);
        void accept(ASTVisitor& visitor) override;
};
class CallStatNode : public StatNode {
public:
    std::string funcName;
    std::vector<std::shared_ptr<ExprNode>> args;
    CallStatNode(const std::string& funcName, std::vector<std::shared_ptr<ExprNode>> args);
    void accept(ASTVisitor& visitor) override;
};

class IfNode : public StatNode {
public:
    std::shared_ptr<ExprNode> cond;
    std::shared_ptr<BlockNode> thenBlock;
    std::shared_ptr<BlockNode> elseBlock; // optional
    IfNode(std::shared_ptr<ExprNode> condition,
           std::shared_ptr<BlockNode> thenBlock,
           std::shared_ptr<BlockNode> elseBlock = nullptr)
        : cond(std::move(condition)),
          thenBlock(std::move(thenBlock)),
          elseBlock(std::move(elseBlock)) {}
    void accept(ASTVisitor& visitor) override;
};

class LoopNode : public StatNode {
public:
    std::shared_ptr<BlockNode> body;
    std::shared_ptr<ExprNode> cond; // optional (for while)
    LoopNode(std::shared_ptr<BlockNode> body,
             std::shared_ptr<ExprNode> cond= nullptr)
        : body(std::move(body)), cond(std::move(cond)) {}
    void accept(ASTVisitor& visitor) override;
};

class BlockNode : public ASTNode {
    public:
        std::vector<std::shared_ptr<DecNode>> decs;
        std::vector<std::shared_ptr<StatNode>> stats;

        BlockNode(
            std::vector<std::shared_ptr<DecNode>> declarations,
            std::vector<std::shared_ptr<StatNode>> statements
        )
            : decs(std::move(declarations)), stats(std::move(statements)) {}
        void accept(ASTVisitor& visitor) override;
};
class FileNode : public ASTNode {
public:
    std::vector<std::shared_ptr<ASTNode>> stats;
    explicit FileNode(std::vector<std::shared_ptr<ASTNode>> stats) : stats(std::move(stats)) {}
    void accept(ASTVisitor& visitor) override;
};

class ProcedureNode : public ASTNode {
public:
    std::string name;
    std::vector<VarInfo> params;
    CompleteType returnType; // optional
    std::shared_ptr<BlockNode> body;
    ProcedureNode(
        const std::string& name,
        const std::vector<VarInfo>& params,
        CompleteType returnType,
        std::shared_ptr<BlockNode> body
    );
    void accept(ASTVisitor& visitor) override;
};

class TypeAliasNode : public ASTNode {
public:
    std::string aliasName;
    TypeAliasNode(const std::string& aliasName, const CompleteType& type);
    void accept(ASTVisitor& visitor) override;
};

class TupleTypeAliasNode : public ASTNode {
public:
    std::string aliasName;
    TupleTypeAliasNode(const std::string& aliasName, CompleteType tupleType);
    void accept(ASTVisitor& visitor) override;
};

class TupleLiteralNode : public ExprNode {
public:
    std::vector<std::shared_ptr<ExprNode>> elements;
    TupleLiteralNode(std::vector<std::shared_ptr<ExprNode>> elements);
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
    std::shared_ptr<ExprNode> expr;
    TypeCastNode(const std::string& targetType, std::shared_ptr<ExprNode> expr);
    void accept(ASTVisitor& visitor) override;
};

class TupleTypeCastNode : public ExprNode {
public:
    std::shared_ptr<ExprNode> expr;
    TupleTypeCastNode(CompleteType targetTupleType, std::shared_ptr<ExprNode> expr);
    void accept(ASTVisitor& visitor) override;
};

class RealNode : public LiteralExprNode {
public:
    double value;
    explicit RealNode(double value);
    void accept(ASTVisitor& visitor) override;
};

