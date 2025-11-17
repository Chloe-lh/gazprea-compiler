#include "AST.h"
#include "ASTVisitor.h"
#include "Types.h"

// ─────────────────────────────────────────────────────────────
// Accept methods for AST nodes
// ─────────────────────────────────────────────────────────────

// File structure
void FileNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void BlockNode::accept(ASTVisitor &visitor) { visitor.visit(this); }

// Expressions
void ParenExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }
void UnaryExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }
void ExpExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }
void MultExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }
void AddExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }
void CompExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }
void NotExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }
void EqExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }
void AndExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }
void OrExpr::accept(ASTVisitor &visitor) { visitor.visit(this); }

// Basic types
void TrueNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void FalseNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void CharNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void IdNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void IntNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void RealNode::accept(ASTVisitor &visitor) { visitor.visit(this); }

// Declarations
void TupleTypedDecNode::accept(ASTVisitor &visitor) { visitor.visit(this); }

void TypedDecNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void InferredDecNode::accept(ASTVisitor &visitor) { visitor.visit(this); }

// Functions
void FuncStatNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void FuncPrototypeNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void FuncBlockNode::accept(ASTVisitor &visitor) { visitor.visit(this); }

// Statements
void CallStatNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void AssignStatNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void OutputStatNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void InputStatNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void BreakStatNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void ContinueStatNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void ReturnStatNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void IfNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void LoopNode::accept(ASTVisitor &visitor) { visitor.visit(this); }

// Extended nodes
void ProcedureBlockNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void TypeAliasNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void TupleTypeAliasNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void TupleLiteralNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void TupleAccessNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void TypeCastNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
void TupleTypeCastNode::accept(ASTVisitor &visitor) { visitor.visit(this); }

// ─────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────

// Expressions
UnaryExpr::UnaryExpr(const std::string &op, std::shared_ptr<ExprNode> operand)
    : UnaryExprNode(op, std::move(operand)) {}

ExpExpr::ExpExpr(const std::string &op, std::shared_ptr<ExprNode> left,
                 std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

MultExpr::MultExpr(const std::string &op, std::shared_ptr<ExprNode> left,
                   std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

AddExpr::AddExpr(const std::string &op, std::shared_ptr<ExprNode> left,
                 std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

CompExpr::CompExpr(const std::string &op, std::shared_ptr<ExprNode> left,
                   std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

NotExpr::NotExpr(const std::string &op, std::shared_ptr<ExprNode> operand)
    : UnaryExprNode(op, std::move(operand)) {}

AndExpr::AndExpr(const std::string &op, std::shared_ptr<ExprNode> left,
                 std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

OrExpr::OrExpr(const std::string &op, std::shared_ptr<ExprNode> left,
               std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}
EqExpr::EqExpr(const std::string &op, std::shared_ptr<ExprNode> left,
               std::shared_ptr<ExprNode> right)
    : BinaryExprNode(op, std::move(left), std::move(right)) {}

ParenExpr::ParenExpr(std::shared_ptr<ExprNode> expr) : expr(std::move(expr)) {}
CharNode::CharNode(char v) : value(v) {}
IntNode::IntNode(int v) : value(v) {}
RealNode::RealNode(double value) : value(value) {}
StringNode::StringNode(std::string v) : value(std::move(v)) {
  this->type = CompleteType(BaseType::STRING);
}
void StringNode::accept(ASTVisitor &visitor) { visitor.visit(this); }
IdNode::IdNode(const std::string &id) : id(id) {}

CallExprNode::CallExprNode(const std::string &name,
                           std::vector<std::shared_ptr<ExprNode>> args)
    : funcName(name), args(std::move(args)) {}

void CallExprNode::accept(ASTVisitor &v) { v.visit(this); }

// FuncCallExpr accept implementation (visitor expects FuncCallExpr*)
void FuncCallExpr::accept(ASTVisitor &v) { v.visit(this); }

// Function nodes
FuncStatNode::FuncStatNode(const std::string &name,
                           const std::vector<VarInfo> &parameters,
                           CompleteType returnType,
                           std::shared_ptr<StatNode> returnStat)
    : FuncNode(name, parameters, std::move(returnType), nullptr,
               std::move(returnStat)) {}

FuncPrototypeNode::FuncPrototypeNode(const std::string &name,
                                     const std::vector<VarInfo> &parameters,
                                     CompleteType returnType)
    : FuncNode(name, parameters, std::move(returnType), nullptr, nullptr) {}

FuncBlockNode::FuncBlockNode(const std::string &name,
                             const std::vector<VarInfo> &parameters,
                             CompleteType returnType,
                             std::shared_ptr<BlockNode> body)
    : FuncNode(name, parameters, std::move(returnType), std::move(body),
               nullptr) {}

// Extended nodes
ProcedureBlockNode::ProcedureBlockNode(const std::string &name,
                             const std::vector<VarInfo> &params,
                             CompleteType returnType,
                             std::shared_ptr<BlockNode> body)
    : name(name), params(params), returnType(std::move(returnType)),
      body(std::move(body)) {}

TypeAliasNode::TypeAliasNode(const std::string &aliasName,
                             const CompleteType &type)
    : aliasName(aliasName) {
  this->type = type;
}

TypeAliasDecNode::TypeAliasDecNode(const std::string &aliasName,
                                   const CompleteType &aliasedType)
    : alias(aliasName) {
  this->type = aliasedType;
}

void TypeAliasDecNode::accept(ASTVisitor &visitor) { visitor.visit(this); }

TupleTypedDecNode::TupleTypedDecNode(const std::string &name,
                                     const std::string &qualifier,
                                     CompleteType tupleType)
    : qualifier(qualifier), init(nullptr) {
  this->name = name;
  this->type = std::move(tupleType);
}

TypedDecNode::TypedDecNode(const std::string &name,
                           std::shared_ptr<TypeAliasNode> type_alias,
                           const std::string &qualifier,
                           std::shared_ptr<ExprNode> init)
    : qualifier(qualifier), type_alias(std::move(type_alias)),
      init(std::move(init)) {
  this->name = name;
}

InferredDecNode::InferredDecNode(const std::string &name,
                                 const std::string &qualifier,
                                 std::shared_ptr<ExprNode> init)
    : qualifier(qualifier), init(std::move(init)) {
  this->name = name;
}

TupleTypeAliasNode::TupleTypeAliasNode(const std::string &aliasName,
                                       CompleteType tupleType)
    : aliasName(aliasName) {
  this->type = tupleType;
}

TupleLiteralNode::TupleLiteralNode(
    std::vector<std::shared_ptr<ExprNode>> elements)
    : elements(std::move(elements)) {}

TupleAccessNode::TupleAccessNode(const std::string &tupleName, int index)
    : tupleName(tupleName), index(index) {}

TypeCastNode::TypeCastNode(const CompleteType &targetType,
               std::shared_ptr<ExprNode> expr)
  : targetType(targetType), expr(std::move(expr)) {}

TupleTypeCastNode::TupleTypeCastNode(const CompleteType &targetTupleType,
                   std::shared_ptr<ExprNode> expr)
  : targetTupleType(targetTupleType), expr(std::move(expr)) {}

// Statements
ReturnStatNode::ReturnStatNode(std::shared_ptr<ExprNode> expr)
    : expr(std::move(expr)) {}
AssignStatNode::AssignStatNode(const std::string &name,
                               std::shared_ptr<ExprNode> expr)
    : name(name), expr(std::move(expr)) {}
OutputStatNode::OutputStatNode(std::shared_ptr<ExprNode> expr)
    : expr(std::move(expr)) {}
