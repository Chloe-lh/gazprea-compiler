#include "AST.h"
#include "ASTVisitor.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

// ─────────────────────────────────────────────────────────────
// Accept methods for AST nodes
// ─────────────────────────────────────────────────────────────

// File structure
void FileNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void BlockNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// Expressions
void ParenExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }
void UnaryExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }
void ExpExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }
void MultExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }
void AddExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }
void CompExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }
void NotExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }
void EqExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }
void AndExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }
void OrExpr::accept(ASTVisitor& visitor) { visitor.visit(this); }

// Basic types
void TrueNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FalseNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void CharNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void IdNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void IntNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void RealNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// Declarations
void TupleDecNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void TypedDecNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void InferredDecNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// Functions
void FuncBlockNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FuncBlockTupleReturnNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FuncStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FuncPrototypeNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FuncPrototypeTupleReturnNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// Statements
void CallStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void AssignStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void OutputStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void InputStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void BreakStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void ContinueStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void ReturnStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void IfNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void LoopNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// Extended nodes
void ProcedureNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void TypeAliasNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void TupleTypeAliasNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void TupleLiteralNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void TupleAccessNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void TypeCastNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void TupleTypeCastNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// ─────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────

// Expressions
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

ParenExpr::ParenExpr() : ExprNode() {}
CharNode::CharNode(char v) : value(v) {}
IntNode::IntNode(int v) : value(v) {}
RealNode::RealNode(double value) : value(value) {}
IdNode::IdNode(const std::string& id) : id(id) {}

// Declarations
TupleDecNode::TupleDecNode(const std::string& name, std::vector<std::string>& types)
    : DecNode(), elementTypes(types) { this->name = name; }

// Function nodes
FuncBlockNode::FuncBlockNode(const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& parameters,
    const std::string& returnType,
    std::unique_ptr<BlockNode> body)
    : FuncNode(name, parameters, returnType, nullptr, std::move(body), nullptr) {}

FuncBlockTupleReturnNode::FuncBlockTupleReturnNode(
    const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& parameters,
    std::unique_ptr<TupleDecNode> returnTupleType,
    std::unique_ptr<BlockNode> body
)
: FuncNode(name, parameters, "", std::move(returnTupleType), std::move(body), nullptr) {}

FuncStatNode::FuncStatNode(
    const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& parameters,
    const std::string& returnType,
    std::unique_ptr<StatNode> returnStat
)
: FuncNode(name, parameters, returnType, nullptr, nullptr, std::move(returnStat)) {}

FuncPrototypeNode::FuncPrototypeNode(const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& parameters,
    const std::string& returnType)
    : FuncNode(name, parameters, returnType, nullptr, nullptr, nullptr) {}

FuncPrototypeTupleReturnNode::FuncPrototypeTupleReturnNode(const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& parameters,
    std::unique_ptr<TupleDecNode> returnTupleType)
    : FuncNode(name, parameters, "", std::move(returnTupleType), nullptr, nullptr) {}

// Extended nodes
ProcedureNode::ProcedureNode(const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& params,
    std::unique_ptr<BlockNode> body)
    : name(name), params(params), body(std::move(body)) {}

TypeAliasNode::TypeAliasNode(const std::string& aliasName, const std::string& typeName)
    : aliasName(aliasName), typeName(typeName) {}

TupleTypeAliasNode::TupleTypeAliasNode(const std::string& aliasName, std::unique_ptr<TupleDecNode> tupleType)
    : aliasName(aliasName), tupleType(std::move(tupleType)) {}

TupleLiteralNode::TupleLiteralNode(std::vector<std::unique_ptr<ExprNode>> elements)
    : elements(std::move(elements)) {}

TupleAccessNode::TupleAccessNode(const std::string& tupleName, int index)
    : tupleName(tupleName), index(index) {}

TypeCastNode::TypeCastNode(const std::string& targetType, std::unique_ptr<ExprNode> expr)
    : targetType(targetType), expr(std::move(expr)) {}

TupleTypeCastNode::TupleTypeCastNode(std::unique_ptr<TupleDecNode> targetTupleType, std::unique_ptr<ExprNode> expr)
    : targetTupleType(std::move(targetTupleType)), expr(std::move(expr)) {}