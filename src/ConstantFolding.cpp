#include "ConstantFolding.h"
#include "AST.h"
#include "CompileTimeExceptions.h"
#include "Types.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <optional>
#include <variant>

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

/*
Deep equality on ConstantValue trees. Used for folding equality on composite constant expressions (arrays/vectors/matrices/tuples/structs).
*/ 
static bool equalConstantValues(const ConstantValue &a, const ConstantValue &b) {
    if (a.type.baseType != b.type.baseType) return false;
    BaseType bt = a.type.baseType;

    // Composite types: recurse element-wise
    if (bt == BaseType::ARRAY ||
        bt == BaseType::VECTOR ||
        bt == BaseType::MATRIX ||
        bt == BaseType::TUPLE  ||
        bt == BaseType::STRUCT) {

        if (!std::holds_alternative<std::vector<ConstantValue>>(a.value) ||
            !std::holds_alternative<std::vector<ConstantValue>>(b.value)) {
            return false;
        }
        const auto &va = std::get<std::vector<ConstantValue>>(a.value);
        const auto &vb = std::get<std::vector<ConstantValue>>(b.value);
        if (va.size() != vb.size()) return false;
        for (size_t i = 0; i < va.size(); ++i) {
            if (!equalConstantValues(va[i], vb[i])) return false;
        }
        return true;
    }

    // Scalar cases
    switch (bt) {
        case BaseType::INTEGER:
            return std::get<int64_t>(a.value) == std::get<int64_t>(b.value);
        case BaseType::REAL:
            return std::get<double>(a.value) == std::get<double>(b.value);
        case BaseType::BOOL:
            return std::get<bool>(a.value) == std::get<bool>(b.value);
        case BaseType::STRING:
            return std::get<std::string>(a.value) == std::get<std::string>(b.value);
        case BaseType::CHARACTER:
            return std::get<char>(a.value) == std::get<char>(b.value);
        default:
            return false;
    }
}
std::optional<std::vector<long double>> extractVectorFromConst(const ConstantValue &cv, bool &hasReal) {
    if (!std::holds_alternative<std::vector<ConstantValue>>(cv.value)) return std::nullopt;
    const auto &elems = std::get<std::vector<ConstantValue>>(cv.value);
    std::vector<long double> out;
    out.reserve(elems.size());
    hasReal = false;
    for (const auto &e : elems) {
        if (e.type.baseType == BaseType::INTEGER) {
            out.push_back(static_cast<long double>(std::get<int64_t>(e.value)));
        } else if (e.type.baseType == BaseType::REAL) {
            out.push_back(static_cast<long double>(std::get<double>(e.value)));
            hasReal = true;
        } else {
            return std::nullopt; // not a numeric literal -> cannot fold
        }
    }
    return out;
}
std::optional<std::vector<std::vector<long double>>> extractMatrixFromConst(const ConstantValue &cv, bool &hasReal) {
    if (!std::holds_alternative<std::vector<ConstantValue>>(cv.value)) return std::nullopt;
    const auto &rows = std::get<std::vector<ConstantValue>>(cv.value);
    std::vector<std::vector<long double>> out;
    out.reserve(rows.size());
    hasReal = false;
    for (const auto &rowCv : rows) {
        bool rowHasReal = false;
        auto rowOpt = extractVectorFromConst(rowCv, rowHasReal);
        if (!rowOpt) return std::nullopt;
        if (rowHasReal) hasReal = true;
        out.push_back(std::move(*rowOpt));
    }
    // optional: check rectangular shape (all rows same length)
    if (!out.empty()) {
        size_t c = out[0].size();
        for (auto &r : out) if (r.size() != c) return std::nullopt;
    }
    return out;
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

    // TODO add support for booleans?
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
    return std::nullopt;
}

void ConstantFoldingVisitor::setConstInCurrentScope(const std::string &ident, const ConstantValue &cv) {
    if (scopes_.empty()) pushScope();
    scopes_.back()[ident] = cv;
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
void ConstantFoldingVisitor::visit(ArrayStrideExpr *node) {}
void ConstantFoldingVisitor::visit(ArraySliceExpr *node) {}
void ConstantFoldingVisitor::visit(ArrayAccessNode *node) {} // not needed

void ConstantFoldingVisitor::visit(ArrayTypedDecNode *node) {
    // Visit the initializer so its constant (if any) is computed. If this
    // declaration is not a `var` and the initializer folded to a compile-time
    // ConstantValue, record it in the current scope under the declared id
    // so later Id lookups will find the constant.
    if (node->init) node->init->accept(*this);
    if (node->qualifier != "var") {
        if (node->init && node->init->constant.has_value()) {
            setConstInCurrentScope(node->id, node->init->constant.value());
        }
    }
}
void ConstantFoldingVisitor::visit(ExprListNode *node) {
    if(!node) return;
    if(!node->list.empty()){
        for(auto &e : node->list){
            e->accept(*this);
        }
    }
}
void ConstantFoldingVisitor::visit(ArrayLiteralNode *node) {
    if (!node) return;
    if (!node->list || node->list->list.empty()) {
        // Represent empty array literal as an ARRAY with UNKNOWN element
        // subtype so it can act as a wildcard for later type promotion.
        node->type = CompleteType(BaseType::ARRAY, CompleteType(BaseType::UNKNOWN), {0});
        return;
    }

    std::vector<ConstantValue> elems;
    bool allConst = true;
    for (auto &e : node->list->list) {
        if (e) e->accept(*this);
        if (!e || !e->constant.has_value()) { allConst = false; break; }
        elems.push_back(e->constant.value());
    }

    // compute common element type using promotion only if all elements constant
    if (allConst) {
        CompleteType common = elems[0].type;
        for (size_t i = 1; i < elems.size(); ++i) {
            CompleteType et = elems[i].type;
            CompleteType promoted = promote(et, common);
            if (promoted.baseType == BaseType::UNKNOWN) promoted = promote(common, et);
            if (promoted.baseType == BaseType::UNKNOWN) {
                throw LiteralError(node->line, "Semantic Analysis: incompatible element types in array literal.");
            }
            common = promoted;
        }
        ConstantValue cv;
        cv.type = node->type;
        cv.value = elems;
        node->constant = cv;
        node->type = cv.type; // keep AST type consistent
        // Record concrete dimension for this array literal so downstream
        // lowering can allocate fixed-size memrefs when emitting literals.
        // Don't overwrite dims if semantic analysis already set them (e.g., for 2D arrays)
        if (node->type.dims.empty()) {
            node->type.dims = { static_cast<int>(elems.size()) };
        }
    }
}
void ConstantFoldingVisitor::visit(RangeExprNode *node) {}
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
void ConstantFoldingVisitor::visit(TupleTypedDecNode *node){
    if (node->init) node->init->accept(*this);
}
void ConstantFoldingVisitor::visit(StructTypedDecNode *node){
    if (node->init) node->init->accept(*this);
}
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
    if (!node) throw std::runtime_error("ConstantFolding::TupleAccessAssignStatNode: null node");
    if (node->target) node->target->accept(*this);
    if (node->expr) node->expr->accept(*this);
    if (node->target) {
        // conservative: forget any constant associated with the tuple variable
        removeConst(node->target->tupleName);
    }
  }

  void ConstantFoldingVisitor::visit(StructAccessAssignStatNode *node) {
    if (!node) throw std::runtime_error("ConstantFolding::StructAccessAssignStatNode: null node");

    if (node->target) node->target->accept(*this);
    if (node->expr) node->expr->accept(*this);
    if (node->target) removeConst(node->target->structName);
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
  // This handles constant folding for all function/struct-constructor calls
void ConstantFoldingVisitor::visit(FuncCallExprOrStructLiteral *node) {
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
/*
compile time constants are stored in the constant field as a Constant Value, arrays will have a std::vector<ConstantValue>
*/
void ConstantFoldingVisitor::visit(DotExpr *node) {
    if (node->left)  node->left->accept(*this);
    if (node->right) node->right->accept(*this);

    // std::cerr << "[CF] visit DotExpr line " << node->line << " op='" << node->op << "'\n";

    if (!node->left || !node->right) {
        std::cerr << "[CF]  children missing\n";
        return;
    }
    bool lHas = node->left->constant.has_value();
    bool rHas = node->right->constant.has_value();
    // std::cerr << "[CF]  left_const=" << lHas << " right_const=" << rHas << "\n";
    if (!lHas || !rHas) {
        std::cerr << "[CF]  skipping fold: one or more children not constant\n";
        return;
    }

    const ConstantValue &Lcv = node->left->constant.value();
    const ConstantValue &Rcv = node->right->constant.value();

    // --- Vector dot product (scalar result) ---
    if (node->type.baseType == BaseType::INTEGER) {
        bool lHasReal = false, rHasReal = false;
        auto lVecOpt = extractVectorFromConst(Lcv, lHasReal);
        auto rVecOpt = extractVectorFromConst(Rcv, rHasReal);
        if (!lVecOpt || !rVecOpt) return;
        const auto &lVec = *lVecOpt;
        const auto &rVec = *rVecOpt;
        if (lVec.size() != rVec.size()) {
            std::cerr << "[CF]  vector sizes mismatch: " << lVec.size() << " vs " << rVec.size() << "\n";
            return;
        }

        std::cerr << "[CF]  folding vector->int of length=" << lVec.size() << "\n";
        long double acc = 0.0L;
        for (size_t i = 0; i < lVec.size(); ++i) acc += lVec[i] * rVec[i];

        int64_t outv = static_cast<int64_t>(std::llround(acc));
        node->constant = getIntConst(outv);
        std::cerr << "[CF]  folded int result=" << outv << "\n";
        return;
    }

    if (node->type.baseType == BaseType::REAL) {
        bool lHasReal = false, rHasReal = false;
        auto lVecOpt = extractVectorFromConst(Lcv, lHasReal);
        auto rVecOpt = extractVectorFromConst(Rcv, rHasReal);
        if (!lVecOpt || !rVecOpt) return;
        const auto &lVec = *lVecOpt;
        const auto &rVec = *rVecOpt;
        if (lVec.size() != rVec.size()) {
            std::cerr << "[CF]  vector sizes mismatch: " << lVec.size() << " vs " << rVec.size() << "\n";
            return;
        }

        std::cerr << "[CF]  folding vector->real of length=" << lVec.size() << "\n";
        long double acc = 0.0L;
        for (size_t i = 0; i < lVec.size(); ++i) acc += lVec[i] * rVec[i];

        double outv = static_cast<double>(acc);
        node->constant = getRealConst(outv);
        std::cerr << "[CF]  folded real result=" << outv << "\n";
        return;
    }

    // --- Matrix / matmul case: result is an array-of-rows (matrix) ---
    if (node->type.baseType == BaseType::ARRAY || node->type.baseType == BaseType::VECTOR || node->type.baseType == BaseType::MATRIX) {
        bool lHasReal = false, rHasReal = false;
        auto lMatOpt = extractMatrixFromConst(Lcv, lHasReal);
        auto rMatOpt = extractMatrixFromConst(Rcv, rHasReal);
        if (!lMatOpt || !rMatOpt) return;
        const auto &lMat = *lMatOpt;
        const auto &rMat = *rMatOpt;
        if (lMat.empty() || rMat.empty()) return;
        size_t M = lMat.size();
        size_t N = lMat[0].size();
        if (N != rMat.size()) return; // inner dim mismatch
        size_t P = rMat[0].size();

        std::cerr << "[CF]  folding matrix matmul M=" << M << " N=" << N << " P=" << P << "\n";

        // compute numeric result matrix
        std::vector<std::vector<long double>> numericRes(M, std::vector<long double>(P, 0.0L));
        for (size_t i = 0; i < M; ++i) {
            if (lMat[i].size() != N) return; // sanity: non-rectangular
            for (size_t j = 0; j < P; ++j) {
                long double sum = 0.0L;
                for (size_t k = 0; k < N; ++k) sum += lMat[i][k] * rMat[k][j];
                numericRes[i][j] = sum;
            }
        }

        // Decide element type: prefer semantic subtype if present, otherwise fallback to any-real-from-extraction
        bool semanticReal = false;
        if (!node->type.subTypes.empty()) {
            CompleteType st = node->type.subTypes[0];
            if (st.baseType == BaseType::ARRAY && !st.subTypes.empty()) st = st.subTypes[0];
            semanticReal = (st.baseType == BaseType::REAL);
        }
        bool anyReal = semanticReal || lHasReal || rHasReal;

        // Build nested ConstantValue representation: vector<ConstantValue(rows)> where each row is ConstantValue(ARRAY of scalars)
        std::vector<ConstantValue> rowsCv;
        rowsCv.reserve(M);
        for (size_t i = 0; i < M; ++i) {
            std::vector<ConstantValue> rowElems;
            rowElems.reserve(P);
            for (size_t j = 0; j < P; ++j) {
                if (anyReal) rowElems.push_back(getRealConst(static_cast<double>(numericRes[i][j])));
                else rowElems.push_back(getIntConst(static_cast<int64_t>(std::llround(numericRes[i][j]))));
            }
            ConstantValue rowCv;
            CompleteType elemCT = anyReal ? CompleteType(BaseType::REAL) : CompleteType(BaseType::INTEGER);
            rowCv.type = CompleteType(BaseType::ARRAY, std::vector<CompleteType>{elemCT});
            rowCv.value = rowElems;
            rowsCv.push_back(rowCv);
        }

        ConstantValue outCv;
        if (node->type.baseType == BaseType::ARRAY) {
            outCv.type = node->type;
        } else if (node->type.baseType == BaseType::MATRIX) {
            // For MATRIX result, preserve the node type and its dims
            // The element type should be the scalar type from node->type.subTypes[0]
            CompleteType elemCT = (node->type.subTypes.empty() || node->type.subTypes[0].baseType == BaseType::UNKNOWN)
                                   ? (anyReal ? CompleteType(BaseType::REAL) : CompleteType(BaseType::INTEGER))
                                   : node->type.subTypes[0];
            std::cerr << "DEBUG: type:" << toString(elemCT.baseType);
            outCv.type = CompleteType(BaseType::MATRIX, elemCT, node->type.dims);
        } else {
            CompleteType elemCT = anyReal ? CompleteType(BaseType::REAL) : CompleteType(BaseType::INTEGER);
            CompleteType rowCT = CompleteType(BaseType::ARRAY, std::vector<CompleteType>{elemCT});
            outCv.type = CompleteType(BaseType::ARRAY, std::vector<CompleteType>{rowCT});
        }
        outCv.value = rowsCv;
        node->constant = outCv;
        return;
    }
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
        if (!node || !node->operand) return;
        node->operand->accept(*this);
        // require a computed constant and that it's a boolean
        if (!node->operand->constant.has_value()) return;
        const ConstantValue &cv = node->operand->constant.value();
        if (cv.type.baseType != BaseType::BOOL) return;
        bool ov = std::get<bool>(cv.value);
        node->constant = getBoolConst(!ov);
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
    // Array / vector / matrix: element-wise equality, only when both sides
    // are already constants of the same composite type.
    if (L.type.baseType == R.type.baseType &&
        (L.type.baseType == BaseType::ARRAY ||
         L.type.baseType == BaseType::VECTOR ||
         L.type.baseType == BaseType::MATRIX)) {
        bool eqAll = equalConstantValues(L, R);
        setResult(eqAll);
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
    // std::cout << "visiting AND Expr";
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
    if (node->op == "or"){
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
    } else if (node->op == "xor") {
        if (node->left && node->right &&
            node->left->constant.has_value() &&
            node->right->constant.has_value() &&
            node->left->constant->type.baseType == BaseType::BOOL &&
            node->right->constant->type.baseType == BaseType::BOOL) {

            bool lv = std::get<bool>(node->left->constant->value);
            bool rv = std::get<bool>(node->right->constant->value);
            node->constant = getBoolConst(lv != rv); // XOR result
        }
    }

  }
  void ConstantFoldingVisitor::visit(TrueNode *node){ node->constant = getBoolConst(true);}
  void ConstantFoldingVisitor::visit(FalseNode *node){ node->constant = getBoolConst(false);}
  void ConstantFoldingVisitor::visit(CharNode *node){ node->constant = getCharConst(node->value);}
  void ConstantFoldingVisitor::visit(IntNode *node){ 
    // std::cout << "Visiting IntNode at " << node 
    //           << ", value=" << node->value << std::endl;
    node->constant = getIntConst(node->value);
}
  void ConstantFoldingVisitor::visit(IdNode *node){ 
    auto v = lookup(node->id);
    if(v.has_value()) { node->constant = v.value(); }
  }

void ConstantFoldingVisitor::visit(StructAccessNode *node) {
    // Do nothing
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
