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
void TupleTypedDecNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

void TypedDecNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void InferredDecNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// Functions
void FuncStatNode::accept(ASTVisitor& visitor) { visitor.visit(this); }
void FuncPrototypeNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

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
UnaryExpr::UnaryExpr(const std::string& op, std::shared_ptr<ExprNode> operand)
    : UnaryExprNode(op, std::move(operand)) {}

ExpExpr::ExpExpr(const std::string& op, std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

MultExpr::MultExpr(const std::string& op, std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

AddExpr::AddExpr(const std::string& op, std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

CompExpr::CompExpr(const std::string& op, std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

NotExpr::NotExpr(const std::string& op, std::shared_ptr<ExprNode> operand)
    : UnaryExprNode(op, std::move(operand)) {}

AndExpr::AndExpr(const std::string& op, std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

OrExpr::OrExpr(const std::string& op, std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}
EqExpr::EqExpr(const std::string& op, std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

ParenExpr::ParenExpr(std::shared_ptr<ExprNode> expr)
    : expr(std::move(expr)) {}
CharNode::CharNode(char v) : value(v) {}
IntNode::IntNode(int v) : value(v) {}
RealNode::RealNode(double value) : value(value) {}
IdNode::IdNode(const std::string& id) : id(id) {}

// Function nodes
FuncStatNode::FuncStatNode(
    const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& parameters,
    CompleteType returnType,
    std::shared_ptr<StatNode> returnStat
)
: FuncNode(name, parameters, std::move(returnType), nullptr, std::move(returnStat)) {}

FuncPrototypeNode::FuncPrototypeNode(const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& parameters,
    CompleteType returnType)
    : FuncNode(name, parameters, std::move(returnType), nullptr, nullptr) {}

// Extended nodes
ProcedureNode::ProcedureNode(const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& params,
    std::shared_ptr<BlockNode> body)
    : name(name), params(params), body(std::move(body)) {}

TypeAliasNode::TypeAliasNode(const std::string& aliasName, const CompleteType& type)
    : aliasName(aliasName) {this->type = type;}

TupleTypeAliasNode::TupleTypeAliasNode(const std::string& aliasName, CompleteType tupleType)
    : aliasName(aliasName) {this->type = tupleType;}

TupleLiteralNode::TupleLiteralNode(std::vector<std::shared_ptr<ExprNode>> elements)
    : elements(std::move(elements)) {}

TupleAccessNode::TupleAccessNode(const std::string& tupleName, int index)
    : tupleName(tupleName), index(index) {}

TypeCastNode::TypeCastNode(const std::string& targetType, std::shared_ptr<ExprNode> expr)
    : targetType(targetType), expr(std::move(expr)) {}

TupleTypeCastNode::TupleTypeCastNode(CompleteType targetTupleType, std::shared_ptr<ExprNode> expr)
    : expr(std::move(expr)) {this->type = targetTupleType;}

// Statements
ReturnStatNode::ReturnStatNode(std::shared_ptr<ExprNode> expr)
    : expr(std::move(expr)) {}
AssignStatNode::AssignStatNode(const std::string& name, std::shared_ptr<ExprNode> expr)
    : name(name), expr(std::move(expr)) {}
OutputStatNode::OutputStatNode(std::shared_ptr<ExprNode> expr)
    : expr(std::move(expr)) {}
CallStatNode::CallStatNode(const std::string& funcName, std::vector<std::shared_ptr<ExprNode>> args)
    : funcName(funcName), args(std::move(args)) {}
