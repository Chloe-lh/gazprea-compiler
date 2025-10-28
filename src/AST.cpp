#include "AST.h"
#include "ASTVisitor.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

// we could implement toString methods here too for printing tree
void FileNode::accept(ASTVisitor& visitor){ visitor.visit(this); }
void BlockNode::accept(ASTVisitor& visitor){ visitor.visit(this);}

//expressions
void ParenExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }
void UnaryExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }
void ExpExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }
void MultExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }
void AddExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }
void CompExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }
void NotExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }
void EqExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }
void AndExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }
void OrExpr::accept(ASTVisitor& visitor){ visitor.visit(this); }

// basic types
void TrueNode::accept(ASTVisitor& visitor){ visitor.visit(this); }
void FalseNode::accept(ASTVisitor& visitor){ visitor.visit(this); }
void CharNode::accept(ASTVisitor& visitor){ visitor.visit(this); }
void IdNode::accept(ASTVisitor& visitor){ visitor.visit(this); }
void IntNode::accept(ASTVisitor& visitor){ visitor.visit(this); }
void RealNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

//declarations
void TupleDecNode::accept(ASTVisitor& visitor){ visitor.visit(this); }
void TypedDecNode::accept(ASTVisitor& visitor){ visitor.visit(this); }
void InferredDecNode::accept(ASTVisitor& visitor){ visitor.visit(this); }

//functions
void FuncBlockNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FuncBlockTupleReturnNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FuncStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FuncPrototypeNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FuncPrototypeTupleReturnNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

//states
void CallStatNode::accept(ASTVisitor& visitor){ visitor.visit(this);}
void AssignStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void OutputStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void InputStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void BreakStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void ContinueStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void ReturnStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void IfNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void LoopNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// CONSTRUCTORS

// expressions
UnaryExpr::UnaryExpr(const std::string& op, std::unique_ptr<ExprNode> operand)
    : UnaryExprNode(op, std::move(operand)) {}
ExpExpr::ExpExpr(const std::string& op, std::unique_ptr<ExprNode> left, std::unique_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}
MultExpr::MultExpr(const std::string& op, std::unique_ptr<ExprNode> left, std::unique_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}
AddExpr::AddExpr(const std::string& op, std::unique_ptr<ExprNode> left, std::unique_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}
CompExpr::CompExpr(const std::string& op, std::unique_ptr<ExprNode> left, std::unique_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}
NotExpr::NotExpr(const std::string& op, std::unique_ptr<ExprNode> operand)
    : UnaryExprNode(op, std::move(operand)) {}
AndExpr::AndExpr(const std::string& op, std::unique_ptr<ExprNode> left, std::unique_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}
OrExpr::OrExpr(const std::string& op, std::unique_ptr<ExprNode> left, std::unique_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

// declarations
TupleDecNode::TupleDecNode(const std::string& name, std::vector<std::string>& types)
    : DecNode(), elementTypes(types) { this->name = name; }
TypedDecNode::TypedDecNode(
    const std::string& name,
    std::unique_ptr<TypeAliasNode> type_alias,
    const std::string& qualifier,
    std::unique_ptr<ExprNode> init
)
: DecNode(), type_alias(std::move(type_alias)), qualifier(qualifier), init(std::move(init)) { this->name = name; }
InferredDecNode::InferredDecNode(
    const std::string& name,
    const std::string& qualifier,
    std::unique_ptr<ExprNode> init)
    : DecNode(), qualifier(qualifier), init(std::move(init)) { this->name = name; }

//statements
CallStatNode::CallStatNode(const std::string& funcName, std::vector<std::unique_ptr<ExprNode>> args)
    : funcName(funcName), args(std::move(args)) {}


ProcedureNode::ProcedureNode(const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& parameters,
    std::unique_ptr<BlockNode> body)
    : name(name), params(parameters), body(std::move(body)) {}
void ProcedureNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

TypeAliasNode::TypeAliasNode(const std::string& aliasName, const std::string& typeName)
    : aliasName(aliasName), typeName(typeName) {}
void TypeAliasNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

TupleTypeAliasNode::TupleTypeAliasNode(const std::string& aliasName, std::unique_ptr<TupleDecNode> tupleType)
    : aliasName(aliasName), tupleType(std::move(tupleType)) {}
void TupleTypeAliasNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

TupleLiteralNode::TupleLiteralNode(std::vector<std::unique_ptr<ExprNode>> elements)
    : elements(std::move(elements)) {}
void TupleLiteralNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

TupleAccessNode::TupleAccessNode(const std::string& tupleName, int index)
    : tupleName(tupleName), index(index) {}
void TupleAccessNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

TypeCastNode::TypeCastNode(const std::string& targetType, std::unique_ptr<ExprNode> expr)
    : targetType(targetType), expr(std::move(expr)) {}
void TypeCastNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

TupleTypeCastNode::TupleTypeCastNode(std::unique_ptr<TupleDecNode> targetTupleType, std::unique_ptr<ExprNode> expr)
    : targetTupleType(std::move(targetTupleType)), expr(std::move(expr)) {}
void TupleTypeCastNode::accept(ASTVisitor& visitor) { visitor.visit(this); }