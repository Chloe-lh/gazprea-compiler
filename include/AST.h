// Tuple type node: represents a tuple type signature (e.g., tuple(int, char))
#pragma once
#include "Scope.h"
#include "Types.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

struct ConstantValue {
  CompleteType type;

  // now includes array constants
  std::variant<int64_t, double, bool, char, std::string,
               std::vector<ConstantValue>> value;

  ConstantValue() : type(CompleteType(BaseType::UNKNOWN)) {}

  ConstantValue(const CompleteType &t,
                std::variant<int64_t, double, bool, char, std::string,
                             std::vector<ConstantValue>>
                    v)
      : type(t), value(std::move(v)) {}
};

enum class CallType { FUNCTION, PROCEDURE, STRUCT_LITERAL, ARRAY_LITERAL };

// abstract class that is extended by the different passes in the pipeline
class ASTVisitor;
// forward declarations
class ExprNode;
class BlockNode;
class TypeAliasNode;
class TupleAccessNode;
class StructAccessNode;
class ArrayAccessNode;
class RangeExprNode;

class ASTNode { // virtual class
public:
  // type
  virtual ~ASTNode() = default;
  virtual void accept(ASTVisitor &visitor) = 0;
  CompleteType type = CompleteType(BaseType::UNKNOWN);
  int line;
};
// super classes
class DecNode : public ASTNode {
public:
  std::string name;
  std::string declTypeName; // Renamed from 'type' to 'declTypeName'
  std::string qualifier = "const";      // default const; overridden where needed
  std::shared_ptr<ExprNode> init;       // optional initializer shared by declarations
  virtual ~DecNode() = default;
  virtual void accept(ASTVisitor &visitor) = 0;
  // SCOPE
};
class ExprNode : public ASTNode {
public:
  virtual ~ExprNode() = default;
  virtual void accept(ASTVisitor &visitor) = 0;
  // Optionalx compile-time constant annotation. When present, this records the
  // evaluated constant and its type so later passes or tests can inspect it.
  std::optional<ConstantValue> constant;
};
// statements
class StatNode : public ASTNode {
public:
  virtual ~StatNode() = default;
  virtual void accept(ASTVisitor &visitor) = 0;
};
// literal expressions
class LiteralExprNode : public ExprNode {
public:
  // Allow default construction; concrete literal visitors or the semantic pass
  // can set `ASTNode::type` later. Also provide a convenience constructor to
  // initialize the inherited `ASTNode::type` when needed.
  LiteralExprNode() = default;
  explicit LiteralExprNode(const CompleteType &t) { this->type = t; }
  virtual ~LiteralExprNode() = default;
};
// expressions
class UnaryExprNode : public ExprNode {
public:
  std::string op;
  std::shared_ptr<ExprNode> operand;
  UnaryExprNode(const std::string &op, std::shared_ptr<ExprNode> operand)
      : op(op), operand(std::move(operand)) {}
};
//  binary expressions
class BinaryExprNode : public ExprNode {
public:
  std::string op;
  std::shared_ptr<ExprNode> left;
  std::shared_ptr<ExprNode> right;
  BinaryExprNode(const std::string &op, std::shared_ptr<ExprNode> left,
                 std::shared_ptr<ExprNode> right)
      : op(op), left(std::move(left)), right(std::move(right)) {}
};
// functions
class FuncNode : public ASTNode {
public:
  std::string name;
  std::vector<VarInfo> parameters;      //  prototypes may omit identifier
  CompleteType returnType;              // optional
  std::shared_ptr<BlockNode> body;      // optional
  std::shared_ptr<StatNode> returnStat; // optional

  FuncNode(const std::string &name, const std::vector<VarInfo> &parameters,
           CompleteType returnType, std::shared_ptr<BlockNode> body = nullptr,
           std::shared_ptr<StatNode> returnStat = nullptr)
      : name(name), parameters(parameters), returnType(std::move(returnType)),
        body(std::move(body)), returnStat(std::move(returnStat)) {}
};

class FuncStatNode : public FuncNode {
public:
  FuncStatNode(const std::string &name, const std::vector<VarInfo> &parameters,
               CompleteType returnType, std::shared_ptr<StatNode> returnStat);
  void accept(ASTVisitor &visitor) override;
};

class FuncPrototypeNode : public FuncNode {
public:
  FuncPrototypeNode(const std::string &name,
                    const std::vector<VarInfo> &parameters, // args technically
                    CompleteType returnType);
  void accept(ASTVisitor &visitor) override;
};

// Functions with a block body
class FuncBlockNode : public FuncNode {
public:
  FuncBlockNode(const std::string &name, const std::vector<VarInfo> &parameters,
                CompleteType returnType, std::shared_ptr<BlockNode> body);
  void accept(ASTVisitor &visitor) override;
};
class BuiltInFuncNode: public ExprNode{
public:
  std::string funcName;
  std::string id;
  BuiltInFuncNode(const std::string &funcName, const std::string &id) : funcName(std::move(funcName)), id(std::move(id)) {}
  void accept(ASTVisitor &visitor) override;
};
/*    Procedure-related               */
class ProcedureBlockNode : public ASTNode {
public:
  std::string name;
  std::vector<VarInfo> params;
  CompleteType returnType; // optional
  std::shared_ptr<BlockNode> body;
  ProcedureBlockNode(const std::string &name,
                     const std::vector<VarInfo> &params,
                     CompleteType returnType, std::shared_ptr<BlockNode> body);
  void accept(ASTVisitor &visitor) override;
};

class ProcedurePrototypeNode : public ASTNode {
public:
  std::string name;
  std::vector<VarInfo> params;
  CompleteType returnType; // optional
  ProcedurePrototypeNode(const std::string &name,
                         const std::vector<VarInfo> &parameters,
                         CompleteType returnType); // optional return type
  void accept(ASTVisitor &visitor) override;
};

// Arrays
class ArraySliceExpr : public ExprNode {
public:
  std::string id;
  std::shared_ptr<RangeExprNode> range;
  ArraySliceExpr(const std::string &id, std::shared_ptr<RangeExprNode> range)
      : id(id), range(std::move(range)) {}
  void accept(ASTVisitor &visitor) override;
};
class ArrayStrideExpr : public ExprNode {
public:
  std::string id;
  std::shared_ptr<ExprNode> expr;
  ArrayStrideExpr(const std::string &id, std::shared_ptr<ExprNode> expr)
      : id(id), expr(std::move(expr)) {}
  void accept(ASTVisitor &visitor) override;
};
class ArrayAccessNode : public ExprNode {
public:
  std::string id;
  std::shared_ptr<ExprNode> indexExpr;
  std::shared_ptr<ExprNode> indexExpr2;  // For 2D array access
  VarInfo *binding = nullptr;
  ArrayAccessNode(const std::string &id, std::shared_ptr<ExprNode> indexExpr) : id(id), indexExpr(std::move(indexExpr)) {}
  ArrayAccessNode(const std::string &id, std::shared_ptr<ExprNode> i1, std::shared_ptr<ExprNode> i2) : id(id), indexExpr(std::move(i1)), indexExpr2(std::move(i2)) {}
  void accept(ASTVisitor &visitor) override;
};
class ArrayLiteralNode;

class ArrayTypedDecNode : public DecNode {
public:
  std::string id;
  CompleteType type;              // Also holds sizes
  ArrayTypedDecNode(const std::string &q, const std::string &id, CompleteType t)
      : id(id), type(std::move(t)) {
    this->qualifier = q;
  }
  ArrayTypedDecNode(const std::string &q, const std::string &id, CompleteType t,
                    std::shared_ptr<ExprNode> i)
      : id(id), type(std::move(t)) {
    this->qualifier = q;
    this->init = std::move(i);
  }

  void accept(ASTVisitor &visitor) override;
};

class ExprListNode : public ASTNode {
public:
  std::vector<std::shared_ptr<ExprNode>> list;
  ExprListNode(std::vector<std::shared_ptr<ExprNode>> list)
      : list(std::move(list)) {}
  void accept(ASTVisitor &visitor) override;
};
class ArrayLiteralNode : public LiteralExprNode {
public:
  std::shared_ptr<ExprListNode> list; // optional
  ArrayLiteralNode(std::shared_ptr<ExprListNode> list)
      : list(std::move(list)) {}
  void accept(ASTVisitor &visitor) override;
};
// Vectors

//   expr RANGE expr    (start..end)
//   RANGE expr         (..end)    -> start == nullptr
//   expr RANGE         (start..)  -> end == nullptr
class RangeExprNode : public ExprNode {
public:
  std::shared_ptr<ExprNode> start; // nullable
  std::shared_ptr<ExprNode> end;   // nullable
  RangeExprNode(std::shared_ptr<ExprNode> s, std::shared_ptr<ExprNode> e)
      : start(std::move(s)), end(std::move(e)) {}
  void accept(ASTVisitor &visitor) override;
};
// expression classes
class ParenExpr : public ExprNode {
public:
  std::shared_ptr<ExprNode> expr;
  explicit ParenExpr(std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};
class UnaryExpr : public UnaryExprNode {
public:
  UnaryExpr(const std::string &op, std::shared_ptr<ExprNode> operand);
  void accept(ASTVisitor &visitor) override;
};
class ExpExpr : public BinaryExprNode {
public:
  ExpExpr(const std::string &op, std::shared_ptr<ExprNode> l,
          std::shared_ptr<ExprNode> r);
  void accept(ASTVisitor &visitor) override;
};
class MultExpr : public BinaryExprNode {
public:
  MultExpr(const std::string &op, std::shared_ptr<ExprNode> l,
           std::shared_ptr<ExprNode> r);
  void accept(ASTVisitor &visitor) override;
};
class AddExpr : public BinaryExprNode {
public:
  AddExpr(const std::string &op, std::shared_ptr<ExprNode> l,
          std::shared_ptr<ExprNode> r);
  void accept(ASTVisitor &visitor) override;
};
class CompExpr : public BinaryExprNode {
public:
  CompExpr(const std::string &op, std::shared_ptr<ExprNode> l,
           std::shared_ptr<ExprNode> r);
  void accept(ASTVisitor &visitor) override;
};
class NotExpr : public UnaryExprNode {
public:
  NotExpr(const std::string &op, std::shared_ptr<ExprNode> operand);
  void accept(ASTVisitor &visitor) override;
};
class EqExpr : public BinaryExprNode {
public:
  EqExpr(const std::string &op, std::shared_ptr<ExprNode> l,
         std::shared_ptr<ExprNode> r);
  void accept(ASTVisitor &visitor) override;
};
class AndExpr : public BinaryExprNode {
public:
  AndExpr(const std::string &op, std::shared_ptr<ExprNode> l,
          std::shared_ptr<ExprNode> r);
  void accept(ASTVisitor &visitor) override;
};
class OrExpr : public BinaryExprNode {
public:
  OrExpr(const std::string &op, std::shared_ptr<ExprNode> l,
         std::shared_ptr<ExprNode> r);
  void accept(ASTVisitor &visitor) override;
};
class DotExpr : public BinaryExprNode{
  public:
  DotExpr(const std::string &op, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r);
  void accept(ASTVisitor &visitor) override;
};
class TrueNode : public LiteralExprNode {
public:
  bool value;
  TrueNode() : LiteralExprNode(CompleteType(BaseType::BOOL)), value(true) {}
  void accept(ASTVisitor &visitor) override;
};
class FalseNode : public LiteralExprNode {
public:
  bool value;
  FalseNode() : LiteralExprNode(CompleteType(BaseType::BOOL)), value(false) {}
  void accept(ASTVisitor &visitor) override;
};
class CharNode : public LiteralExprNode {
public:
  char value;
  explicit CharNode(char v);
  void accept(ASTVisitor &visitor) override;
};
class IntNode : public LiteralExprNode {
public:
  int value;
  explicit IntNode(int v);
  void accept(ASTVisitor &visitor) override;
};
class IdNode : public ExprNode {
public:
  const std::string id;
  VarInfo *binding = nullptr; // bound VarInfo from semantic analysis
  explicit IdNode(const std::string &id);
  void accept(ASTVisitor &visitor) override;
};

class TypeAliasDecNode : public DecNode {
public:
  std::string alias;
  TypeAliasDecNode(const std::string &alias, const CompleteType &type);
  void accept(ASTVisitor &visitor) override;
};

class TupleTypedDecNode : public DecNode {
public:
  // Constructor matching the usage
  TupleTypedDecNode(const std::string &q, const std::string &name,
                    CompleteType tupleType)
  {
    this->name = name;
    this->type = std::move(tupleType);
    this->qualifier = q;
  }

  void accept(ASTVisitor &visitor) override;
};

// Handles both standalone struct type declarations,
//    e.g. struct MyStruct (integer memb1, boolean memb2);
// As well as struct + variable declarations
//    e.g. struct MyStruct (integer memb1, boolean memb2) varDec;
class StructTypedDecNode : public DecNode {
public:
  StructTypedDecNode(const std::string &name, const std::string &qualifier,
                     CompleteType structType);
  void accept(ASTVisitor &visitor) override;
};

class TypedDecNode : public DecNode {
public:
  std::shared_ptr<TypeAliasNode> type_alias; // type OR alias
  TypedDecNode(const std::string &name,
               std::shared_ptr<TypeAliasNode> type_alias,
               const std::string &qualifier = "",
               std::shared_ptr<ExprNode> init = nullptr);
  std::optional<ConstantValue> constant;
  void accept(ASTVisitor &visitor) override;
};
class InferredDecNode : public DecNode {
public:
  InferredDecNode(const std::string &name, const std::string &qualifier,
                  std::shared_ptr<ExprNode> init);
  void accept(ASTVisitor &visitor) override;
};
// statement nodes
class AssignStatNode : public StatNode {
public:
  std::string name;
  std::shared_ptr<ExprNode> expr;
  AssignStatNode(const std::string &name, std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};
class DestructAssignStatNode : public StatNode {
public:
  std::vector<std::string> names;
  std::shared_ptr<ExprNode> expr;
  DestructAssignStatNode(std::vector<std::string> names,
                         std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};
class ArrayAccessAssignStatNode : public StatNode {
public:
  std::shared_ptr<ArrayAccessNode> target;
  std::shared_ptr<ExprNode> expr;
  ArrayAccessAssignStatNode(std::shared_ptr<ArrayAccessNode> target,
                            std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};
class TupleAccessAssignStatNode : public StatNode {
public:
  std::shared_ptr<TupleAccessNode> target;
  std::shared_ptr<ExprNode> expr;
  TupleAccessAssignStatNode(std::shared_ptr<TupleAccessNode> target,
                            std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};

class StructAccessAssignStatNode : public StatNode {
public:
  std::shared_ptr<StructAccessNode> target;
  std::shared_ptr<ExprNode> expr;
  StructAccessAssignStatNode(std::shared_ptr<StructAccessNode> target,
                             std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};

class OutputStatNode : public StatNode {
public:
  std::shared_ptr<ExprNode> expr;
  OutputStatNode(std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};
class InputStatNode : public StatNode {
public:
  std::string name;
  explicit InputStatNode(const std::string &name) : name(name) {}
  void accept(ASTVisitor &visitor) override;
};
class BreakStatNode : public StatNode {
public:
  BreakStatNode() = default;
  void accept(ASTVisitor &visitor) override;
};
class ContinueStatNode : public StatNode {
public:
  ContinueStatNode() = default;
  void accept(ASTVisitor &visitor) override;
};
class ReturnStatNode : public StatNode {
public:
  std::shared_ptr<ExprNode> expr;
  ReturnStatNode(std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};
// does not return a calue
class CallExprNode : public ExprNode {
public:
  std::string funcName;
  std::vector<std::shared_ptr<ExprNode>> args;
  std::optional<FuncInfo> resolvedFunc;
  CallType callType = CallType::PROCEDURE;
  CallExprNode(const std::string &, std::vector<std::shared_ptr<ExprNode>>);
  void accept(ASTVisitor &v) override;
};
// Expression-style function call OR struct literal node (grammar: ID '(' expr*
// ')')
class FuncCallExprOrStructLiteral : public CallExprNode {

public:
  // Same as CallExprNode constructor except callType is set to
  // `CallType::FUNCTION` Semantic analysis should resolve and update `callType`
  // to `CallType::STRUCT_LITERAL` where applicable
  FuncCallExprOrStructLiteral(const std::string &,
                              std::vector<std::shared_ptr<ExprNode>>);
  void accept(ASTVisitor &v) override;
};

class MethodCallExpr : public ExprNode {
public:
    std::string objectName;
    std::string methodName;
    std::vector<std::shared_ptr<ExprNode>> args;
    MethodCallExpr(const std::string &obj, const std::string &method, std::vector<std::shared_ptr<ExprNode>> args)
        : objectName(obj), methodName(method), args(std::move(args)) {}
    void accept(ASTVisitor &v) override;
};

class MethodCallStatNode : public StatNode {
public:
    std::string objectName;
    std::string methodName;
    std::vector<std::shared_ptr<ExprNode>> args;
    MethodCallStatNode(const std::string &obj, const std::string &method, std::vector<std::shared_ptr<ExprNode>> args)
        : objectName(obj), methodName(method), args(std::move(args)) {}
    void accept(ASTVisitor &v) override;
};

// can be used in statements
class CallStatNode : public StatNode {
public:
  // Wrapper around expression-style call/struct literal expression node
  std::shared_ptr<FuncCallExprOrStructLiteral> call;
  explicit CallStatNode(std::shared_ptr<FuncCallExprOrStructLiteral> c)
      : call(std::move(c)) {}
  void accept(ASTVisitor &v) override;
};

class IfNode : public StatNode {
public:
  std::shared_ptr<ExprNode> cond;
  std::shared_ptr<BlockNode> thenBlock;
  std::shared_ptr<BlockNode> elseBlock;
  std::shared_ptr<StatNode> thenStat;
  std::shared_ptr<StatNode> elseStat;

  // A simple constructor.
  explicit IfNode(std::shared_ptr<ExprNode> condition)
      : cond(std::move(condition)), thenBlock(nullptr), elseBlock(nullptr),
        thenStat(nullptr), elseStat(nullptr) {}

  void accept(ASTVisitor &visitor) override;
};

enum class LoopKind { Plain, While, WhilePost };
// plain : no Condition
// while : condition body
// whilePost : body condition
class LoopNode : public StatNode {
public:
  std::shared_ptr<BlockNode> body;
  std::shared_ptr<ExprNode> cond; // optional (for while)
  LoopKind kind;
  LoopNode(std::shared_ptr<BlockNode> body,
           std::shared_ptr<ExprNode> cond = nullptr)
      : body(std::move(body)), cond(std::move(cond)), kind(LoopKind::Plain) {}
  void accept(ASTVisitor &visitor) override;
};

class BlockNode : public ASTNode {
public:
  std::vector<std::shared_ptr<DecNode>> decs;
  std::vector<std::shared_ptr<StatNode>> stats;

  BlockNode(std::vector<std::shared_ptr<DecNode>> declarations,
            std::vector<std::shared_ptr<StatNode>> statements)
      : decs(std::move(declarations)), stats(std::move(statements)) {}
  void accept(ASTVisitor &visitor) override;
};
class FileNode : public ASTNode {
public:
  std::vector<std::shared_ptr<ASTNode>> stats;
  explicit FileNode(std::vector<std::shared_ptr<ASTNode>> stats)
      : stats(std::move(stats)) {}
  void accept(ASTVisitor &visitor) override;
};

class TypeAliasNode : public ASTNode {
public:
  std::string aliasName;
  TypeAliasNode(const std::string &aliasName, const CompleteType &type);
  void accept(ASTVisitor &visitor) override;
};

class TupleTypeAliasNode : public ASTNode {
public:
  std::string aliasName;
  TupleTypeAliasNode(const std::string &aliasName, CompleteType tupleType);
  void accept(ASTVisitor &visitor) override;
};

class StructAccessNode : public ExprNode {
public:
  std::string structName;
  std::string fieldName;
  size_t fieldIndex;
  VarInfo *binding = nullptr; // bound struct variable from semantic analysis
  StructAccessNode(const std::string &structName, const std::string &fieldName);
  void accept(ASTVisitor &visitor) override;
};

class TupleLiteralNode : public ExprNode {
public:
  std::vector<std::shared_ptr<ExprNode>> elements;
  TupleLiteralNode(std::vector<std::shared_ptr<ExprNode>> elements);
  void accept(ASTVisitor &visitor) override;
};

class TupleAccessNode : public ExprNode {
public:
  std::string tupleName;
  int index;
  VarInfo *binding = nullptr; // bound tuple variable from semantic analysis
  TupleAccessNode(const std::string &tupleName, int index);
  void accept(ASTVisitor &visitor) override;
};

class TypeCastNode : public ExprNode {
public:
  CompleteType targetType;
  std::shared_ptr<ExprNode> expr;
  // If the target type in a cast was written as an identifier (alias),
  // capture the alias name here so semantic analysis can resolve it
  // When this is non-empty, `targetType` may be UNKNOWN as a placeholder.
  std::string targetAliasName;
  TypeCastNode(const CompleteType &targetType, std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};

class TupleTypeCastNode : public ExprNode {
public:
  CompleteType targetTupleType;
  std::shared_ptr<ExprNode> expr;
  TupleTypeCastNode(const CompleteType &targetTupleType,
                    std::shared_ptr<ExprNode> expr);
  void accept(ASTVisitor &visitor) override;
};

class RealNode : public LiteralExprNode {
public:
  double value;
  explicit RealNode(double value);
  void accept(ASTVisitor &visitor) override;
};

class StringNode : public LiteralExprNode {
public:
  std::string value;
  explicit StringNode(std::string v);
  void accept(ASTVisitor &visitor) override;
};
