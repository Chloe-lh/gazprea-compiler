#include "ConstantFolding.h"
#include "AST.h"
#include "Types.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <optional>

/*

  // Root
  void ConstantFoldingVisitor::visit(FileNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}

  // Functions
  void ConstantFoldingVisitor::visit(FuncStatNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(FuncPrototypeNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(FuncBlockNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(ProcedureNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}

  // Declarations
  void ConstantFoldingVisitor::visit(TypedDecNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(InferredDecNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(TupleTypedDecNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(TypeAliasDecNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(TypeAliasNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(TupleTypeAliasNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}

  // Statements
  void ConstantFoldingVisitor::visit(AssignStatNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(OutputStatNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(InputStatNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(BreakStatNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(ContinueStatNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(ReturnStatNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(CallStatNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(IfNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(LoopNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(BlockNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}

  // Expressions
  void ConstantFoldingVisitor::visit(ParenExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(FuncCallExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(UnaryExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(ExpExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(MultExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(AddExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(CompExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(NotExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(EqExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(AndExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(OrExpr *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(TrueNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(FalseNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(CharNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(IntNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(IdNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(TupleLiteralNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(TupleAccessNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");} 
  void ConstantFoldingVisitor::visit(TypeCastNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(TupleTypeCastNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(RealNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
  void ConstantFoldingVisitor::visit(StringNode *node){ throw std::runtime_error("TupleTypeCastNode not implemented");}
*/

static ConstantValue getIntConst(int64_t data) {
    return ConstantValue(CompleteType(BaseType::INTEGER), static_cast<int64_t>(data));
}
static ConstantValue getRealConst(double data) {
    return ConstantValue(CompleteType(BaseType::REAL), static_cast<double>(data));
}
static ConstantValue getBoolConst(bool data) {
    return ConstantValue(CompleteType(BaseType::BOOL), static_cast<bool>(data));
}
static ConstantValue getCharConst(char data) {
    return ConstantValue(CompleteType(BaseType::CHARACTER), static_cast<char>(data));
}
static ConstantValue getStringConst(const std::string &s) {
    return ConstantValue(CompleteType(BaseType::STRING), s);
}

void ConstantFoldingVisitor::debugPrintScopes() const {
    std::cout << "--- Current scopes ---\n";
    int level = 0;
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
        std::cout << "Scope level " << level++ << ":\n";
        for (const auto &kv : *it) {
            const auto &name = kv.first;
            const auto &cv = kv.second;
            std::cout << "  " << name << " = ";
            switch (cv.type.baseType) {
                case BaseType::INTEGER:
                    std::cout << std::get<int64_t>(cv.value);
                    break;
                case BaseType::REAL:
                    std::cout << std::get<double>(cv.value);
                    break;
                case BaseType::BOOL:
                    std::cout << (std::get<bool>(cv.value) ? "true" : "false");
                    break;
                case BaseType::CHARACTER:
                    std::cout << std::get<char>(cv.value);
                    break;
                case BaseType::STRING:
                    std::cout << "\"" << std::get<std::string>(cv.value) << "\"";
                    break;
                default:
                    std::cout << "<unsupported type>";
            }
            std::cout << "\n";
        }
    }
    std::cout << "----------------------\n";
}


// Helper: fold binary numeric operations (+ - * / %)
// Returns an optional ConstantValue when folding is safe, std::nullopt otherwise.
static std::optional<ConstantValue> foldBinaryArithmetic(const ConstantValue &L,
                                                         const ConstantValue &R,
                                                         const std::string &op) {
    // Integer-only fast path
    if (L.type.baseType == BaseType::INTEGER && R.type.baseType == BaseType::INTEGER) {
        int64_t lv = std::get<int64_t>(L.value);
        int64_t rv = std::get<int64_t>(R.value);
        if (op == "+") return getIntConst(lv + rv);
        if (op == "-") return getIntConst(lv - rv);
        if (op == "*") return getIntConst(lv * rv);
        if (op == "/") {
            if (rv == 0) return std::nullopt; // avoid div-by-zero folding
            return getIntConst(lv / rv);
        }
        if (op == "%") {
            if (rv == 0) return std::nullopt;
            return getIntConst(lv % rv);
        }
        return std::nullopt;
    }

    // Numeric promotion to real when either side is REAL (or integer)
    bool leftIsNum = (L.type.baseType == BaseType::REAL || L.type.baseType == BaseType::INTEGER);
    bool rightIsNum = (R.type.baseType == BaseType::REAL || R.type.baseType == BaseType::INTEGER);
    if (leftIsNum && rightIsNum) {
        double ld = (L.type.baseType == BaseType::REAL) ? std::get<double>(L.value)
                                                         : static_cast<double>(std::get<int64_t>(L.value));
        double rd = (R.type.baseType == BaseType::REAL) ? std::get<double>(R.value)
                                                         : static_cast<double>(std::get<int64_t>(R.value));
        if (op == "+") return getRealConst(ld + rd);
        if (op == "-") return getRealConst(ld - rd);
        if (op == "*") return getRealConst(ld * rd);
        if (op == "/") {
            if (rd == 0.0) return std::nullopt; // avoid div-by-zero folding
            return getRealConst(ld / rd);
        }
        //TODO maybe add support for real/int modulo?
        return std::nullopt;
    }

    // TODO add support for booleans
    return std::nullopt;
}
/*SCOPING
    - pushScope() - entering any scope (if, while, func)
        - apply before adding parameter binds
    - popScope() - exiting 
    - SetConstInCurrentscope
        - after visiting a declartion with a constant initializer (const integer)
    - removeConst(name) - on assignment (not dec) with a const
        - look up id inside scope
    - lookup - when visiting an IdNode to see if id has a constant yet
*/

void ConstantFoldingVisitor::pushScope() { scopes_.emplace_back(); }
void ConstantFoldingVisitor::popScope() { if (!scopes_.empty()) scopes_.pop_back(); }

std::optional<ConstantValue> ConstantFoldingVisitor::lookup(const std::string &ident) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
        auto var = it->find(ident);
        if (var != it->end()) {
            return var->second;
        }
    }
    std::cout << "No constant found for " << ident << std::endl;
    return std::nullopt;
}

void ConstantFoldingVisitor::setConstInCurrentScope(const std::string &ident, const ConstantValue &cv) {
    if (scopes_.empty()) pushScope();
    scopes_.back()[ident] = cv;
    debugPrintScopes();
}

void ConstantFoldingVisitor::removeConst(const std::string &id) {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
        auto &m = *it;
        auto var = m.find(id);
        if (var != m.end()) { m.erase(var); return; }
    }
}

// Root
void ConstantFoldingVisitor::visit(FileNode *node){
    // Ensure we start with a clean scope stack for each file visit. This
    // is defensive in case a visitor instance is (re)used; it prevents
    // leftover constant bindings from a previous run affecting the current
    // pass.
    scopes_.clear();
    // initialize scope
    pushScope();
    // traverse each node and call ConstantFoldingVisitor::visit on each node
    for(auto &n : node->stats){
        if(n) n->accept(*this);
    }
    popScope();
}

// Functions
void ConstantFoldingVisitor::visit(FuncStatNode *node){ 
    pushScope();
    // parameters should be resolved in Semantic pass
    if(node->returnStat) node->returnStat->accept(*this);
    popScope();
}
void ConstantFoldingVisitor::visit(FuncPrototypeNode *node){}
void ConstantFoldingVisitor::visit(FuncBlockNode *node){ 
    pushScope();
    // parameters should be resolved in Semantic pass
    if(node->body) node->body->accept(*this);
    popScope();
}
void ConstantFoldingVisitor::visit(ProcedureBlockNode *node) {
    pushScope();     // new scope for parameters + locals

    // Parameters are represented as VarInfo in the AST and do not have
    // associated AST nodes to visit here. Any parameter constants (if
    // applicable) are recorded elsewhere during declaration handling, so
    // nothing to do for each param in the folding pass.

    // Visit the body
    if (node->body) {
        node->body->accept(*this);
    }

    popScope();      // leave procedure scope
}

void ConstantFoldingVisitor::visit(ProcedurePrototypeNode *node) {
    // No body to fold; nothing to do for prototypes.
}

// Declarations
void ConstantFoldingVisitor::visit(TypedDecNode *node){ 
    if(node->type_alias) node->type_alias->accept(*this);
    if(node->init) node->init->accept(*this);
    // check for const
    if(node->qualifier != "var"){ //const -> set Const In Current Scope
        // if const and init has a compile time value -> record
        if(node->init && node->init->constant.has_value()){
            setConstInCurrentScope(node->name, node->init->constant.value());
            node->constant = node->init->constant;
        }
    } 
}
void ConstantFoldingVisitor::visit(InferredDecNode *node){ 
    // init
    if(node->init) node->init->accept(*this);
    // check for const
    if(node->qualifier != "var"){ //const -> set Const In Current Scope
        // if const and init has a compile time value -> record
        if(node->init && node->init->constant.has_value()){
            setConstInCurrentScope(node->name, node->init->constant.value());
        }
    } 
}
  void ConstantFoldingVisitor::visit(TupleTypedDecNode *node){ if (node->init) node->init->accept(*this); }
  void ConstantFoldingVisitor::visit(TypeAliasDecNode *node){} 
  void ConstantFoldingVisitor::visit(TypeAliasNode *node){}
  void ConstantFoldingVisitor::visit(TupleTypeAliasNode *node){}

  // Statements
  void ConstantFoldingVisitor::visit(AssignStatNode *node){ 
    if(node->expr){
        node->expr->accept(*this);
        removeConst(node->name); // reassign state
    }
  }
  void ConstantFoldingVisitor::visit(DestructAssignStatNode *node){
    if (node->expr) {
        node->expr->accept(*this);
        for (const auto &name : node->names) {
            removeConst(name);
        }
    }
  }
  void ConstantFoldingVisitor::visit(TupleAccessAssignStatNode *node){
    if (!node) return;
    if (node->target) node->target->accept(*this);
    if (node->expr) node->expr->accept(*this);
    if (node->target) {
        // conservative: forget any constant associated with the tuple variable
        removeConst(node->target->tupleName);
    }
  }
  void ConstantFoldingVisitor::visit(OutputStatNode *node){ if(node->expr) node->expr->accept(*this);}
  void ConstantFoldingVisitor::visit(InputStatNode *node){}
  void ConstantFoldingVisitor::visit(BreakStatNode *node){}
  void ConstantFoldingVisitor::visit(ContinueStatNode *node){}
  void ConstantFoldingVisitor::visit(ReturnStatNode *node){ if(node->expr) node->expr->accept(*this);}
  void ConstantFoldingVisitor::visit(CallStatNode *node){ if(node->call) node->call->accept(*this); } //forwards toFuncCallExpr
  void ConstantFoldingVisitor::visit(IfNode *node){
    // Evaluate condition first so its .constant may be available
    if (node->cond) node->cond->accept(*this);

    // If condition is a compile-time boolean, only visit the taken branch
    // should help performance
    if (node->cond && node->cond->constant.has_value() && node->cond->constant->type.baseType == BaseType::BOOL) {
        bool condVal = std::get<bool>(node->cond->constant->value);
        if (condVal) {
            if (node->thenBlock) node->thenBlock->accept(*this);
            else if (node->thenStat) node->thenStat->accept(*this);
        } else {
            if (node->elseBlock) node->elseBlock->accept(*this);
            else if (node->elseStat) node->elseStat->accept(*this);
        }
        return;
    }
    // Otherwise conservatively visit both sides (no extra scope push here;
    // BlockNodes create their own scopes when appropriate).
    if (node->thenBlock) node->thenBlock->accept(*this);
    else if (node->thenStat) node->thenStat->accept(*this);

    if (node->elseBlock) node->elseBlock->accept(*this);
    else if (node->elseStat) node->elseStat->accept(*this);
  }
  void ConstantFoldingVisitor::visit(LoopNode *node){
    switch(node->kind){
        case LoopKind::Plain:
            //just has a body (no condition)
            if(node->body) node->body->accept(*this);
            break;
        case LoopKind::While:
            if(node->cond) node->cond->accept(*this);
            if(node->cond && node->cond->constant.has_value()){ //condition has a compile time value
                bool condVal = std::get<bool>(node->cond->constant->value);
                if(condVal) if(node->body) node->body->accept(*this);
                break;
            }
            if(node->body) node->body->accept(*this);
            break;
        case LoopKind::WhilePost:
            if(node->body) node->body->accept(*this);
            if(node->cond) node->cond->accept(*this);
            break;
    }
  }
  void ConstantFoldingVisitor::visit(BlockNode *node){
    pushScope();
    // Visit declarations first (they are in the declarations area of a block)
    for (auto &dec : node->decs) {
        if (dec) dec->accept(*this);
    }
    // Then visit statements in order
    for (auto &stat : node->stats) {
        if (stat) stat->accept(*this);
    }
    popScope();
  }

  // Expressions
  void ConstantFoldingVisitor::visit(ParenExpr *node){ 
    if(node->expr){
        node->expr->accept(*this);
        // assign value if available
        if (node->expr->constant.has_value()) node->constant = node->expr->constant;
    }
  }
  //this handles constant folding for all functions
void ConstantFoldingVisitor::visit(FuncCallExpr *node) {
  // 1) Visit args so their .constant gets computed
  for (auto &a : node->args) if (a) a->accept(*this);

  // 2) collect constants
  bool allConst = true;
  std::vector<ConstantValue> argVals;
  for (auto &a : node->args) {
    if (!a || !a->constant.has_value()) { allConst = false; break; }
    argVals.push_back(a->constant.value());
  }
  if (!allConst) return;

  // 3) whitelist foldable intrinsics (example)
    if (node->funcName == "len" && argVals.size() == 1
            && argVals[0].type.baseType == BaseType::STRING) {
        const auto &s = std::get<std::string>(argVals[0].value);
    node->constant = getIntConst(static_cast<int64_t>(s.size()));
    return;
  }
    if (node->funcName == "abs" && argVals.size() == 1) {
    auto &cv = argVals[0];
    if (cv.type.baseType == BaseType::INTEGER) {
      int64_t v = std::get<int64_t>(cv.value);
      node->constant = getIntConst(v < 0 ? -v : v);
      return;
    } else if (cv.type.baseType == BaseType::REAL) {
      double v = std::get<double>(cv.value);
      node->constant = getRealConst(std::fabs(v));
      return;
    }
  }
  // else: can't fold; leave node->constant empty
}
void ConstantFoldingVisitor::visit(UnaryExpr *node){
    node->operand->accept(*this);
    if (!node->operand || !node->operand->constant.has_value()) return;
    if(node->op == "==" || node->op == "--"){
        if(auto id = dynamic_cast<IdNode*>(node->operand.get())){
            removeConst(id->id);
        }
        return;
    }
    return;
}
// exponentiation: right-associative expr EXP expr
void ConstantFoldingVisitor::visit(ExpExpr *node) {
    // Visit children first so their constants are computed
    if (node->left) node->left->accept(*this);
    if (node->right) node->right->accept(*this);

    // Both sides must be constant to fold
    if (!node->left || !node->right) return;
    if (!node->left->constant.has_value() || !node->right->constant.has_value())
        return;

    const auto &leftConst = node->left->constant.value();
    const auto &rightConst = node->right->constant.value();

    // INTEGER ^ INTEGER -> integer when exponent >= 0
    if (leftConst.type.baseType == BaseType::INTEGER &&
        rightConst.type.baseType == BaseType::INTEGER) {
        int64_t base = std::get<int64_t>(leftConst.value);
        int64_t exp  = std::get<int64_t>(rightConst.value);

        if (exp < 0) {
            // negative integer exponent would produce non-integer -> skip
            return;
        }
        // fast exponentiation (exponentiation by squaring)
        int64_t result = 1;
        int64_t b = base;
        int64_t e = exp;
        while (e > 0) {
            if (e & 1) result *= b;
            e >>= 1;
            if (e) b *= b;
        }
        node->constant = getIntConst(result);
        return;
    }

    // If either side is REAL (or mixed int/real), compute as real
    if ((leftConst.type.baseType == BaseType::REAL || leftConst.type.baseType == BaseType::INTEGER) &&
        (rightConst.type.baseType == BaseType::REAL || rightConst.type.baseType == BaseType::INTEGER)) {
        double base = (leftConst.type.baseType == BaseType::REAL)
                          ? std::get<double>(leftConst.value)
                          : static_cast<double>(std::get<int64_t>(leftConst.value));
        double exp = (rightConst.type.baseType == BaseType::REAL)
                         ? std::get<double>(rightConst.value)
                         : static_cast<double>(std::get<int64_t>(rightConst.value));

        // Domain checks: 0^negative is undefined (skip folding)
        if (base == 0.0 && exp < 0.0) return;

        double result = std::pow(base, exp);
        node->constant = getRealConst(result);
        return;
    }
    // otherwise: unsupported types -> don't fold
}

  void ConstantFoldingVisitor::visit(MultExpr *node){ 
    // Visit children first so their constants are computed
    if (node->left) node->left->accept(*this);
    if (node->right) node->right->accept(*this);

    // Both sides must be constant to fold
    if (!node->left || !node->right) return;
    if (!node->left->constant.has_value() || !node->right->constant.has_value())
        return;

    const auto &leftConst = node->left->constant.value();
    const auto &rightConst = node->right->constant.value();

    auto folded = foldBinaryArithmetic(leftConst, rightConst, node->op);
    if (folded.has_value()) node->constant = folded.value();
}
void ConstantFoldingVisitor::visit(AddExpr *node){ 
    // Visit children first so their constants are computed
    if (node->left) node->left->accept(*this);
    if (node->right) node->right->accept(*this);

    std::cout << "Visiting AddExpr at " << node 
              << ", left node at " << node->left.get() 
              << ", right node at " << node->right.get() << std::endl;

    // Now check their constants
    if (!node->left->constant.has_value()) 
        std::cout << "Left child constant is NONE" << std::endl;
    if (!node->right->constant.has_value()) 
        std::cout << "Right child constant is NONE" << std::endl;

      // Both sides must be constant to fold
    if (!node->left || !node->right) return;
    if (!node->left->constant.has_value() || !node->right->constant.has_value())
        return;

    const auto &leftConst = node->left->constant.value();
    const auto &rightConst = node->right->constant.value();

    auto folded = foldBinaryArithmetic(leftConst, rightConst, node->op);
    if (folded.has_value()) node->constant = folded.value();
}
  void ConstantFoldingVisitor::visit(CompExpr *node){ 
    if (node->left) node->left->accept(*this);
    if (node->right) node->right->accept(*this);
    // conservative: only fold if both constants and simple numeric/comparison
    if (!node->left || !node->right) return;
    if (!node->left->constant.has_value() || !node->right->constant.has_value()) return;
    const ConstantValue &L = node->left->constant.value();
    const ConstantValue &R = node->right->constant.value();
    // Only handle numeric comparisons here (int or real). Otherwise skip.
    bool leftIsNum = (L.type.baseType == BaseType::REAL || L.type.baseType == BaseType::INTEGER);
    bool rightIsNum = (R.type.baseType == BaseType::REAL || R.type.baseType == BaseType::INTEGER);
    if (!leftIsNum || !rightIsNum) return;

    double lv = (L.type.baseType == BaseType::REAL) ? std::get<double>(L.value):static_cast<double>(std::get<int64_t>(L.value));
    double rv = (R.type.baseType == BaseType::REAL) ? std::get<double>(R.value):static_cast<double>(std::get<int64_t>(R.value));

    bool res = false;
    if (node->op == ">") res = lv > rv;
    else if (node->op == "<") res = lv < rv;
    else if (node->op == ">=") res = lv >= rv;
    else if (node->op == "<=") res = lv <= rv;

    node->constant = getBoolConst(res);
  }
  void ConstantFoldingVisitor::visit(NotExpr *node){ 


  }
void ConstantFoldingVisitor::visit(EqExpr *node){
    // Evaluate children first
    if (node->left) node->left->accept(*this);
    if (node->right) node->right->accept(*this);

    if (!node->left || !node->right) return;
    if (!node->left->constant.has_value() || !node->right->constant.has_value()) return;

    const ConstantValue &L = node->left->constant.value();
    const ConstantValue &R = node->right->constant.value();

    // Helper to set result (handles == and !=)
    auto setResult = [&](bool eq){
        bool res = (node->op == "!=") ? !eq : eq;
        node->constant = getBoolConst(res);
    };
    // Numeric (int/real) comparison: promote to double
    bool Lnum = (L.type.baseType == BaseType::INTEGER || L.type.baseType == BaseType::REAL);
    bool Rnum = (R.type.baseType == BaseType::INTEGER || R.type.baseType == BaseType::REAL);
    if (Lnum && Rnum) {
        double lv = (L.type.baseType == BaseType::REAL) ? std::get<double>(L.value) : static_cast<double>(std::get<int64_t>(L.value));
        double rv = (R.type.baseType == BaseType::REAL) ? std::get<double>(R.value) : static_cast<double>(std::get<int64_t>(R.value));
        setResult(lv == rv);
        return;
    }
    // Bool
    if (L.type.baseType == BaseType::BOOL && R.type.baseType == BaseType::BOOL) {
        setResult(std::get<bool>(L.value) == std::get<bool>(R.value));
        return;
    }
    // String
    if (L.type.baseType == BaseType::STRING && R.type.baseType == BaseType::STRING) {
        setResult(std::get<std::string>(L.value) == std::get<std::string>(R.value));
        return;
    }
    // Character
    if (L.type.baseType == BaseType::CHARACTER && R.type.baseType == BaseType::CHARACTER) {
        setResult(std::get<char>(L.value) == std::get<char>(R.value));
        return;
    }
    // Otherwise: conservative, don't fold mixed/unsupported comparisons
    return;
}
void ConstantFoldingVisitor::visit(AndExpr *node){
    // Evaluate children first so their constants (if any) are computed
    if (node->left) node->left->accept(*this);
    if (node->right) node->right->accept(*this);

    // Only fold AND when it's safe respecting left-to-right evaluation:
    //  - if left is compile-time false -> whole expression is false (short-circuit)
    //  - else if left is compile-time true and right is compile-time bool -> result is right
    //  - otherwise we can't fold
    if (node->left && node->left->constant.has_value() && node->left->constant->type.baseType == BaseType::BOOL) {
        bool lv = std::get<bool>(node->left->constant->value);
        if (!lv) {
            node->constant = getBoolConst(false);
            return;
        }
        // left is true; only fold if right is a compile-time bool
        if (node->right && node->right->constant.has_value() && node->right->constant->type.baseType == BaseType::BOOL) {
            bool rv = std::get<bool>(node->right->constant->value);
            node->constant = getBoolConst(rv);
        }
    }
}
void ConstantFoldingVisitor::visit(OrExpr *node){ 
    // Evaluate children first so their constants (if any) are computed
    if (node->left) node->left->accept(*this);
    if (node->right) node->right->accept(*this);

    // Short-circuit-safe OR folding (left-to-right):
    // - if left is compile-time true -> whole expression is true
    // - else if left is compile-time false and right is compile-time bool -> result is right
    // - otherwise cannot fold safely
    if (node->left && node->left->constant.has_value() && node->left->constant->type.baseType == BaseType::BOOL) {
        bool lv = std::get<bool>(node->left->constant->value);
        if (lv) {
            node->constant = getBoolConst(true);
            return;
        }
        // left is false; fold only if right is a compile-time bool
        if (node->right && node->right->constant.has_value() && node->right->constant->type.baseType == BaseType::BOOL) {
            bool rv = std::get<bool>(node->right->constant->value);
            node->constant = getBoolConst(rv);
        }
    }
  }
  void ConstantFoldingVisitor::visit(TrueNode *node){ node->constant = getBoolConst(true);}
  void ConstantFoldingVisitor::visit(FalseNode *node){ node->constant = getBoolConst(false);}
  void ConstantFoldingVisitor::visit(CharNode *node){ node->constant = getCharConst(node->value);}
  void ConstantFoldingVisitor::visit(IntNode *node){ 
    std::cout << "Visiting IntNode at " << node 
              << ", value=" << node->value << std::endl;
    node->constant = getIntConst(node->value);
}
  void ConstantFoldingVisitor::visit(IdNode *node){ 
    auto v = lookup(node->id);
    if(v.has_value()) { node->constant = v.value(); }
  }
  void ConstantFoldingVisitor::visit(TupleLiteralNode *node){ 
    for(auto &e: node->elements){
    if(e) e->accept(*this);} //this is will set each elements .constant
  }
void ConstantFoldingVisitor::visit(TupleAccessNode *node){ } //not needed
void ConstantFoldingVisitor::visit(TypeCastNode *node){ 
    // Visit inner expression first
    if (node->expr) node->expr->accept(*this);
    if (!node->expr) return;
    if (!node->expr->constant.has_value()) return;

    const ConstantValue &cv = node->expr->constant.value();
    BaseType from = cv.type.baseType;
    BaseType to = node->targetType.baseType;

    // Only handle scalar-to-scalar casts here (bool,char,int,real)
    if (!canScalarCast(from, to)) return;

    // Perform explicit scalar conversions conservatively
    if (to == BaseType::INTEGER) {
        if (from == BaseType::INTEGER) { node->constant = cv; return; }
        if (from == BaseType::REAL) { node->constant = getIntConst(static_cast<int64_t>(std::get<double>(cv.value))); return; }
        if (from == BaseType::BOOL) { node->constant = getIntConst(std::get<bool>(cv.value) ? 1 : 0); return; }
        if (from == BaseType::CHARACTER) { node->constant = getIntConst(static_cast<int64_t>(std::get<char>(cv.value))); return; }
        return;
    }
    if (to == BaseType::REAL) {
        if (from == BaseType::REAL) { node->constant = cv; return; }
        if (from == BaseType::INTEGER) { node->constant = getRealConst(static_cast<double>(std::get<int64_t>(cv.value))); return; }
        if (from == BaseType::BOOL) { node->constant = getRealConst(std::get<bool>(cv.value) ? 1.0 : 0.0); return; }
        if (from == BaseType::CHARACTER) { node->constant = getRealConst(static_cast<double>(std::get<char>(cv.value))); return; }
        return;
    }
    if (to == BaseType::BOOL) {
        if (from == BaseType::BOOL) { node->constant = cv; return; }
        if (from == BaseType::INTEGER) { node->constant = getBoolConst(std::get<int64_t>(cv.value) != 0); return; }
        if (from == BaseType::REAL) { node->constant = getBoolConst(std::get<double>(cv.value) != 0.0); return; }
        if (from == BaseType::CHARACTER) { node->constant = getBoolConst(std::get<char>(cv.value) != '\0'); return; }
        return;
    }
    if (to == BaseType::CHARACTER) {
        if (from == BaseType::CHARACTER) { node->constant = cv; return; }
        if (from == BaseType::INTEGER) { node->constant = getCharConst(static_cast<char>(std::get<int64_t>(cv.value))); return; }
        if (from == BaseType::REAL) { node->constant = getCharConst(static_cast<char>(static_cast<int64_t>(std::get<double>(cv.value)))); return; }
        if (from == BaseType::BOOL) { node->constant = getCharConst(std::get<bool>(cv.value) ? 1 : 0); return; }
        return;
    }
    // For other target types (string, tuple, etc.) be conservative and do not fold
    return;
}
void ConstantFoldingVisitor::visit(TupleTypeCastNode *node){
    // Visit the inner expression to compute any inner constants, but we
    // do not yet support tuple-shaped ConstantValue in the AST constant
    // representation. So accept the child but do not fold the cast.
    if (node->expr) node->expr->accept(*this);
    return;
}
void ConstantFoldingVisitor::visit(RealNode *node){ node->constant = getRealConst(node->value);}
void ConstantFoldingVisitor::visit(StringNode *node){ node->constant = getStringConst(node->value);}
