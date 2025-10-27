#include "AST.h"
#include "ASTVisitor.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

ExprNode::ExprNode(){
    // typing
}
// we could implement toString methods here too for printing tree
void FileNode::accept(ASTVisitor& visitor){ visitor.visit(this)}

//expressions
void ParenExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}
void UnaryExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}
void ExpExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}
void MultExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}
void AddExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}
void CompExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}
void NotExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}
void EqExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}
void AndExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}
void OrExpr::accept(ASTVisitor& visitor){ visitor.visit(this)}

// basic types
void TrueNode::accept(ASTVisitor& visitor){ visitor.visit(this)}
void FalseNode::accept(ASTVisitor& visitor){ visitor.visit(this)}
void CharNode::accept(ASTVisitor& visitor){ visitor.visit(this)}
void IdNode::accept(ASTVisitor& visitor){ visitor.visit(this)}
void IntNode::accept(ASTVisitor& visitor){ visitor.visit(this)}

//declarations
void TupleDecNode::accept(ASTVisitor& visitor){ visitor.visit(this)}
void TypedDecNode::accept(ASTVisitor& visitor){ visitor.visit(this)}
void InferredDecNode::accept(ASTVisitor& visitor){ visitor.visit(this)}

// CONSTRUCTORS

// expressions
UnaryExpr::UnaryExpr(std::string op, std::unique_ptr<ExprNode> operand)
    : op(op), operand(std::move(operand)) {}
ExpExpr::ExpExpr(std::string op, std::unique_ptr<ExprNode> left, std::unqiue_ptr<ExprNode> right)
    : op(op) left(std::move(left)) right(std::move(right)) {}
MultExpr::MultExpr(std::string op, std::unique_ptr<ExprNode> left, std::unqiue_ptr<ExprNode> right)
    : op(op) left(std::move(left)) right(std::move(right)) {}
AddExpr::AddExpr(std::string op, std::unique_ptr<ExprNode> left, std::unqiue_ptr<ExprNode> right)
    : op(op) left(std::move(left)) right(std::move(right)) {}
CompExpr::CompExpr(std::string op, std::unique_ptr<ExprNode> left, std::unqiue_ptr<ExprNode> right)
    : op(op) left(std::move(left)) right(std::move(right)) {}
NotExpr::NotExpr(std::string op, std::unique_ptr<ExprNode> operand)
    : op(op), operand(std::move(operand)) {}
AndExpr::AndExpr(std::string op, std::unique_ptr<ExprNode> left, std::unqiue_ptr<ExprNode> right)
    : op(op) left(std::move(left)) right(std::move(right)) {}
OrExpr::OrExpr(std::string op, std::unique_ptr<ExprNode> left, std::unqiue_ptr<ExprNode> right)
    : op(op) left(std::move(left)) right(std::move(right)) {}

// declarations
TupleDecNode::TupleDecNode(const std::string& name, std::vector<std::string>& types)
    : name(name), elementTypes(types) {}
TypedDecNode::TypedDecNode(
    const std::string& name,
    const std::string& type_alias,
    std::unique_ptr<ExprNode> init = nullptr,
    const std::string& qualifier = ""
) : qualifier(qualifier), name(name), type_alias(type_alias), init(std::move(init)) {}
InferredDecNode::InferredDecNode(
    const std::string& name,
    const std::string& qualifier,
    std::unique_ptr<ExprNode> init)
    : name(name), qualifier(qualifier), init(std::move(init)) {}

//statements
CallStatNode::CallStatNode(
    const std::string& name,
    std::vecotr<std::unique_ptr<ExprNode>> args)
    : name(name), args(args) {}