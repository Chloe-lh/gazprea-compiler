#include "SemanticAnalysisVisitor.h"
#include "CompileTimeExceptions.h"
#include "Types.h"
#include "run_time_errors.h"
#include <stdexcept>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <set>


// Resolve a CompleteType whose baseType is UNRESOLVED using the current scope's
// global type alias table. Throws bug if UNRESOLVED and no alias name provided
CompleteType resolveUnresolvedType(Scope *scope, const CompleteType &t) {
    if (!scope) {
        return t;
    }

    // First, resolve this level if it's an unresolved alias.
    CompleteType result = t;
    if (result.baseType == BaseType::UNRESOLVED) {
        if (result.aliasName.empty()) {
            throw std::runtime_error(
                "Semantic Analysis: encountered UNRESOLVED type with no alias name.");
        }
        // Scope::resolveAlias will throw SymbolError if the alias is not defined.
        CompleteType *aliased = scope->resolveAlias(result.aliasName);
        result = *aliased;
    }

    // Recursively normalise any composite subtypes as well.
    if (!result.subTypes.empty()) {
        for (auto &sub : result.subTypes) {
            sub = resolveUnresolvedType(scope, sub);
        }
    }

    return result;
}

Scope* SemanticAnalysisVisitor::getRootScope() {
    return this->root_.get();
}

void SemanticAnalysisVisitor::visit(FileNode* node) {
    // Init and enter global scope
    // TODO: handle type aliases here
    // note: can shadow other symbol names
    scopeByCtx_.clear();
    current_ = nullptr;
    enterScopeFor(node, false, nullptr);
    current_->setGlobalTrue();

    for (const auto& stat: node->stats) {
        stat->accept(*this);
    }

    exitScope();

    // Ensure main exists
    if (!seenMain_) {
        throw GlobalError(1, "Semantic Analysis: procedure main() not defined.");
    }
}

/* TODO insert line number for error
*/
void SemanticAnalysisVisitor::visit(FuncStatNode* node) {
    std::vector<VarInfo> params;
    params.reserve(node->parameters.size());
    std::unordered_set<std::string> paramNames;
    for (size_t i = 0; i < node->parameters.size(); ++i) {
        const auto& v = node->parameters[i];
        VarInfo param = v;
        if (v.identifier.empty()) {
            // Should not happen
            throw std::runtime_error("Semantic Analysis: FATAL: parameter name required in function definition '" + node->name + "'.");
        }
        if (!paramNames.insert(v.identifier).second) {
            throw SymbolError(1, "Semantic Analysis: duplicate parameter name '" + v.identifier + "' in function '" + node->name + "'.");
        }
        param.type = resolveUnresolvedType(current_, param.type);
        params.push_back(param);
    }

    // Push resolved types onto params (handling type aliasing)
    for (size_t i = 0; i < node->parameters.size() && i < params.size(); ++i) {
        node->parameters[i].type = params[i].type;
    }

    // Resolve alias in return type, if any.
    node->returnType = resolveUnresolvedType(current_, node->returnType);

    try {
        current_->declareFunc(node->name, params, node->returnType);
    } catch (...) {
        // If already declared, ensure it resolves to the same signature
        FuncInfo* existing = current_->resolveFunc(node->name, params);
        if (existing->funcReturn != node->returnType) {
            throw std::runtime_error("Semantic Analysis: conflicting return type for function '" + node->name + "'.");
        }
    }

    // Enter function scope, bind parameters
    enterScopeFor(node, false, &node->returnType);
    current_->setInFunctionTrue(); 
    for (auto& v : params) {
        if (!v.isConst) { throw AssignError(1, "Non-const arg given as function parameter");}
        current_->declareVar(v.identifier, v.type, true);
    }

    // One-liner must be a return statement, should not throw error bc handled by grammar
    auto ret = std::dynamic_pointer_cast<ReturnStatNode>(node->returnStat);
    if (!ret) {
        throw std::runtime_error("Semantic Analysis: FATAL: single-statement function '" + node->name + "' must be a return statement.");
    }
    ret->accept(*this);

    exitScope();
}

void SemanticAnalysisVisitor::visit(TypedDecNode* node) {
    if (!current_->isDeclarationAllowed()) {
        throw DefinitionError(1, "Semantic Analysis: Declarations must appear at the top of a block."); // FIXME: placeholder error
    }
    
    // Visit the type alias first to resolve any type aliases
    if (node->type_alias) {
        node->type_alias->accept(*this);
    }
    
    if (node->init) {
        node->init->accept(*this);
    }

    bool isConst = true;
    if (node->qualifier == "var") {
        if (current_->isInGlobal()) { throw GlobalError(1, "Cannot use var in global"); }
        isConst = false;
    } else if (node->qualifier == "const") {
    } else {
        throw std::runtime_error("Semantic Analysis: Invalid qualifier provided for typed declaration '" + node->qualifier + "'.");
    }

    // Declared type is carried as a CompleteType on the alias node
    node->type_alias->accept(*this);
    CompleteType varType = resolveUnresolvedType(current_, node->type_alias->type);

    // Ensure not already declared in scope
    current_->declareVar(node->name, varType, isConst);

    // Ensure init expr type matches with var type (if provided)
    if (node->init != nullptr) {
        handleAssignError(node->name, varType, node->init->type);
    }

    node->type_alias->type = varType;
    node->type = varType;
}


/* TODO add error line number */
void SemanticAnalysisVisitor::visit(FuncPrototypeNode* node) {
    // Convert parameter list to VarInfo (names may be empty for prototypes)
    std::vector<VarInfo> params;
    params.reserve(node->parameters.size());
    for (size_t i = 0; i < node->parameters.size(); ++i) {
        // name may be empty or different for prototypes
        const auto& v = node->parameters[i];
        VarInfo param = v;
        param.type = resolveUnresolvedType(current_, param.type);
        params.push_back(param);
    }

    // Propagate resolved parameter types back to the AST node
    for (size_t i = 0; i < node->parameters.size() && i < params.size(); ++i) {
        node->parameters[i].type = params[i].type;
    }

    node->returnType = resolveUnresolvedType(current_, node->returnType);

    // Declare the function signature in the current (global) scope
    // Function prototypes may omit param names so we check that
    try {
        current_->declareFunc(node->name, params, node->returnType);
    } catch (...) {
        FuncInfo* existing = current_->resolveFunc(node->name, params);
        if (existing->funcReturn != node->returnType) {
            throw SymbolError(1, "Semantic Analysis: conflicting return type for function prototype '" + node->name + "'.");
        }
    }
}

/* TODO add line numbers */
void SemanticAnalysisVisitor::visit(FuncBlockNode* node) {
    // Build parameter VarInfos (const by default)
    std::vector<VarInfo> params;
    params.reserve(node->parameters.size());
    std::unordered_set<std::string> paramNames;
    for (size_t i = 0; i < node->parameters.size(); ++i) {
        const auto& v = node->parameters[i];
        VarInfo param = v;
        if (v.identifier.empty()) { 
            // should not happen
            throw std::runtime_error("Semantic Analysis: FATAL: parameter name required in function definition '" + node->name + "'.");
        }
        if (!paramNames.insert(v.identifier).second) {
            throw SymbolError(1, "Semantic Analysis: duplicate parameter name '" + v.identifier + "' in function '" + node->name + "'.");
        }
        param.type = resolveUnresolvedType(current_, param.type);
        params.push_back(param);
    }

    // Propagate resolved parameter types back to the AST node
    for (size_t i = 0; i < node->parameters.size() && i < params.size(); ++i) {
        node->parameters[i].type = params[i].type;
    }

    node->returnType = resolveUnresolvedType(current_, node->returnType);

    // Declare or validate existing prototype declr
    try {
        current_->declareFunc(node->name, params, node->returnType);
    } catch (...) {
        FuncInfo* existing = current_->resolveFunc(node->name, params);
        if (existing->funcReturn != node->returnType) {
            throw std::runtime_error("Semantic Analysis: conflicting return type for function '" + node->name + "'.");
        }
    }

    // Enter function scope, bind parameters
    enterScopeFor(node, false, &node->returnType);
    current_->setInFunctionTrue(); 
    for (auto& v : params) {
        if (!v.isConst) { throw AssignError(1, "Non-const arg given as function parameter");}
        current_->declareVar(v.identifier, v.type, true);
    }

    // Analyze body
    if (!node->body) {
        throw std::runtime_error("Semantic Analysis: FATAL: function '" + node->name + "' missing body.");
    }
    node->body->accept(*this);

    // Ensure all paths return
    if (!guaranteesReturn(node->body.get())) {
        throw ReturnError(1, "Semantic Analysis: not all control paths return in function '" + node->name + "'.");
    }

    exitScope();
}

/* TODO add error line numbers */
void SemanticAnalysisVisitor::visit(ProcedureBlockNode* node) {
    // Resolve any alias-based return type first
    node->returnType = resolveUnresolvedType(current_, node->returnType);

    // Special case: main() constraints
    if (node->name == "main") {
        if (seenMain_) {
            throw SymbolError(1, "Semantic Analysis: Multiple definitions of procedure main().");
        }
        seenMain_ = true;
        if (!node->params.empty()) {
            throw MainError(1, "Semantic Analysis: procedure main() must not take parameters.");
        }
        if (node->returnType.baseType != BaseType::INTEGER) {
            throw MainError(1, "Incorrect return type for main procedure");
        }
    }

    // Build parameter VarInfos, default const. 
    // TODO: handle 'var' once AST carries it
    std::vector<VarInfo> params;
    params.reserve(node->params.size());
    std::unordered_set<std::string> paramNames;
    for (size_t i = 0; i < node->params.size(); ++i) {
        const auto& v = node->params[i];
        VarInfo param = v;
        if (v.identifier.empty()) {
            // should not happen
            throw std::runtime_error("Semantic Analysis:FATAL: parameter name required in procedure '" + node->name + "'.");
        }
        if (!paramNames.insert(v.identifier).second) {
            throw SymbolError(1, std::string("Semantic Analysis: duplicate parameter name '") + v.identifier + "' in procedure '" + node->name + "'.");
        }
        param.type = resolveUnresolvedType(current_, param.type);
        params.push_back(param);
    }

    // Propagate resolved parameter types back to the AST node
    for (size_t i = 0; i < node->params.size() && i < params.size(); ++i) {
        node->params[i].type = params[i].type;
    }

    // Declare or validate existing declaration
    try {
        current_->declareProc(node->name, params, node->returnType);
    } catch (...) {
        ProcInfo* existing = current_->resolveProc(node->name, params);
        if (existing->procReturn != node->returnType) {
            throw TypeError(1, "Semantic Analysis: conflicting return type for procedure '" + node->name + "'.");
        }
    }

    // Enter procedure scope, bind params
    enterScopeFor(node, false, &node->returnType);
    for (auto& v : params) {
        current_->declareVar(v.identifier, v.type, v.isConst);
    }

    if (!node->body) {
        // should not happen
        throw std::runtime_error("Semantic Analysis: FATAL: procedure '" + node->name + "' missing body.");
    }
    node->body->accept(*this);

    // If non-void return expected, ensure all paths return
    if (node->returnType.baseType != BaseType::UNKNOWN) {
        if (!guaranteesReturn(node->body.get())) {
            throw ReturnError(1, "Semantic Analysis: not all control paths return in procedure '" + node->name + "'.");
        }
    }

    exitScope();
}

void SemanticAnalysisVisitor::visit(InferredDecNode* node) {
    if (!current_->isDeclarationAllowed()) {
        throw DefinitionError(1, "Semantic Analysis: Declarations must appear at the top of a block."); // FIXME: placeholder error
    }
    node->init->accept(*this);

    bool isConst = true;
    if (node->qualifier == "var") {
        isConst = false;
    } else if (node->qualifier == "const") {
    } else {
        throw std::runtime_error("Semantic Analysis: Invalid qualifier provided for type inference '" + node->qualifier + "'.");
    }

    CompleteType varType = resolveUnresolvedType(current_, node->init->type); // no need to check promotability


    // Ensure not already declared in scope
    current_->declareVar(node->name, varType, isConst);

    node->type = varType;
}

void SemanticAnalysisVisitor::visit(TupleTypedDecNode* node) {
    if (!current_->isDeclarationAllowed()) {
        throw DefinitionError(1, "Semantic Analysis: Declarations must appear at the top of a block."); // FIXME: placeholder error
    }
    if (node->init) {
        node->init->accept(*this);
    }
    // For tuple-typed declarations, the declared type is already present
    // on the declaration node as a CompleteType
    CompleteType varType = resolveUnresolvedType(current_, node->type);

    // const by default
    bool isConst = true;
    if (node->qualifier == "var") {
        isConst = false;
    } else if (node->qualifier == "const" || node->qualifier.empty()) {
    } else {
        throw std::runtime_error(
            "Semantic Analysis: Invalid qualifier provided for tuple declaration '" +
            node->qualifier + "'.");
    }

    // Ensure not already declared in scope
    current_->declareVar(node->name, varType, isConst);

    // Ensure init expr type matches with var type (if provided)
    if (node->init != nullptr) {
        handleAssignError(node->name, varType, node->init->type);
    }

    node->type = varType;
}

void SemanticAnalysisVisitor::visit(TypeAliasDecNode* node) {
    if (!current_->isInGlobal()) {
        throw StatementError(1, "Alias declaration in non-global scope '" + node->alias + "'.");
    }
    // Resolve aliased type. Support aliasing built-ins or another alias.
    CompleteType aliased = node->type;
    if (aliased.baseType == BaseType::UNKNOWN && !node->declTypeName.empty()) {
        // Try resolving as another alias name
        try {
            aliased = *current_->resolveAlias(node->declTypeName);
        } catch (...) {
            // If not an alias, leave as UNKNOWN; builder should have set to built-in type
        }
    }
    current_->declareAlias(node->alias, aliased);

    // assume node has been initialized with correct type if not UNKNOWN
}

void SemanticAnalysisVisitor::visit(TypeAliasNode *node) {
    if (node->aliasName != "") {
        node->type = *current_->resolveAlias(node->aliasName);
    }

    // if no alias, assume node already initialized with correct type
}

void SemanticAnalysisVisitor::visit(TupleTypeAliasNode *node) {
    // This node represents a declaration: TYPEALIAS tuple_dec ID
    if (!current_->isInGlobal()) {
        throw StatementError(1, "Alias declaration in non-global scope '" + node->aliasName + "'.");
    }
    current_->declareAlias(node->aliasName, node->type);
}

void SemanticAnalysisVisitor::visit(AssignStatNode* node) {
    node->expr->accept(*this);

    // handles if undeclared
    const VarInfo* varInfo = current_->resolveVar(node->name);
    
    if (varInfo->isConst) {
        throw AssignError(1, "Semantic Analysis: cannot assign to const variable '" + node->name + "'."); // TODO add line num
    }

    CompleteType varType = resolveUnresolvedType(current_, varInfo->type);
    CompleteType exprType = resolveUnresolvedType(current_, node->expr->type);
    handleAssignError(node->name, varType, exprType);

    node->type = varType;
}

void SemanticAnalysisVisitor::visit(DestructAssignStatNode* node) {
    if (!node->expr) {
        throw AssignError(1, "Semantic Analysis: missing expression in destructuring assignment.");
    }

    // Analyse RHS first to know its tuple shape
    node->expr->accept(*this);
    CompleteType rhsType = resolveUnresolvedType(current_, node->expr->type);

    if (rhsType.baseType != BaseType::TUPLE) {
        throw AssignError(1, "Semantic Analysis: destructuring assignment requires a tuple expression on the right-hand side.");
    }

    if (rhsType.subTypes.size() != node->names.size()) {
        throw AssignError(1, "Semantic Analysis: tuple arity mismatch in destructuring assignment.");
    }

    // Check each target variable
    for (size_t i = 0; i < node->names.size(); ++i) {
        const std::string &name = node->names[i];
        VarInfo* varInfo = current_->resolveVar(name);
        if (!varInfo) {
            throw SymbolError(1, "Semantic Analysis: variable '" + name + "' not defined in destructuring assignment.");
        }
        if (varInfo->isConst) {
            throw AssignError(1, "Semantic Analysis: cannot assign to const variable '" + name + "' in destructuring assignment.");
        }

        CompleteType varType = resolveUnresolvedType(current_, varInfo->type);
        CompleteType elemType = resolveUnresolvedType(current_, rhsType.subTypes[i]);
        handleAssignError(name, varType, elemType);
    }

    node->type = CompleteType(BaseType::UNKNOWN);
}

void SemanticAnalysisVisitor::visit(OutputStatNode* node) {
    node->expr->accept(*this); // handle expr
    node->type = CompleteType(BaseType::UNKNOWN); // streams do not have a type
}

void SemanticAnalysisVisitor::visit(InputStatNode* node) {
    // checks must be performed at runtime due to input ambiguity 

    node->type = CompleteType(BaseType::UNKNOWN); // streams do not have a type
}

void SemanticAnalysisVisitor::visit(BreakStatNode* node) {
    if (!current_->isInLoop()) {
        throw StatementError(1, "Cannot use 'break' outside of loop."); // TODO add line num
    }
}

void SemanticAnalysisVisitor::visit(ContinueStatNode* node) {
    if (!current_->isInLoop()) {
        throw StatementError(1, "Cannot use 'continue' outside of loop."); // TODO add line num
    }
}

void SemanticAnalysisVisitor::visit(ReturnStatNode* node) {
    // Allow return inside func/proc, determined by non-null expected return type on the scope
    if (current_->getReturnType() == nullptr) {
        throw StatementError(1, "Cannot use 'return' outside of function."); // TODO add line num
    }

    // If expression provided, type-check against expected return type
    if (node->expr) {
        node->expr->accept(*this);
        handleAssignError("", *current_->getReturnType(), node->expr->type);
    } else {
        // No value returned: only legal if the declared return type is 'void' equivalent
        if (current_->getReturnType()->baseType != BaseType::UNKNOWN) {
            throw TypeError(1, "Semantic Analysis: Non-void return required by declaration.");
        }
    }
}

void SemanticAnalysisVisitor::visit(CallStatNode* node) {
    // The CallStatNode wraps an expression-style FuncCallExpr in `call`.
    if (!node->call) {
        throw std::runtime_error("Semantic Analysis: FATAL: empty call statement");
    }
    if (current_->isInFunction()) {
        throw CallError(1, "Cannot call procedure inside a function.");
    }

    std::vector<VarInfo> args;
    args.reserve(node->call->args.size());
    for (const auto& e : node->call->args) {
        if (e) e->accept(*this);
        CompleteType argType = e ? resolveUnresolvedType(current_, e->type)
                                 : CompleteType(BaseType::UNKNOWN);
        args.push_back(VarInfo{"", argType, true});
    }

    // Resolve as procedure only. prevent calling a function via 'call'
    try {
        (void) current_->resolveProc(node->call->funcName, args);
    } catch (...) {
        throw SymbolError(1, "Semantic Analysis: Unknown procedure '" + node->call->funcName + "' in call statement.");
    }

    // Statements have no resultant type
    node->type = CompleteType(BaseType::UNKNOWN);
}

void SemanticAnalysisVisitor::visit(TupleAccessAssignStatNode* node) {
    if (!node->target || !node->expr) {
        throw AssignError(1, "Semantic Analysis: malformed tuple access assignment.");
    }

    // Analyse LHS tuple access first (binds tuple variable + element type)
    node->target->accept(*this);

    if (!node->target->binding) {
        throw std::runtime_error("Semantic Analysis: FATAL: TupleAccessNode missing binding in assignment.");
    }

    VarInfo* tupleVar = node->target->binding;
    if (tupleVar->isConst) {
        throw AssignError(1, "Semantic Analysis: cannot assign to element of const tuple '" +
                                 node->target->tupleName + "'.");
    }

    // visit rhs
    node->expr->accept(*this);

    CompleteType elemType = resolveUnresolvedType(current_, node->target->type);
    CompleteType exprType = resolveUnresolvedType(current_, node->expr->type);

    handleAssignError(node->target->tupleName, elemType, exprType);

    node->type = CompleteType(BaseType::UNKNOWN);
}

void SemanticAnalysisVisitor::visit(FuncCallExpr* node) {
    // Evaluate argument expressions and build a signature to resolve the callee
    std::vector<VarInfo> args;
    args.reserve(node->args.size());
    for (const auto& e : node->args) {
        if (e) e->accept(*this);
        CompleteType argType = e ? resolveUnresolvedType(current_, e->type)
                                 : CompleteType(BaseType::UNKNOWN);
        args.push_back(VarInfo{"", argType, true});
    }

    // Try resolving as function
    // -----------------------
    FuncInfo* finfo = nullptr;
    try {
        finfo = current_->resolveFunc(node->funcName, args);
    } catch (...) {
        finfo = nullptr;
    }
    if (finfo) {
        // Function call in expression
        node->type = finfo->funcReturn;
        node->resolvedFunc = *finfo; // cache resolved info for later passes
        return;
    }

    // Then try resolving as procedure
    // -----------------------
    ProcInfo* pinfo = nullptr;
    try {
        pinfo = current_->resolveProc(node->funcName, args);
    } catch (...) {
        pinfo = nullptr;
    }

    if (pinfo) {
        // Procedures may have a return type; only those may appear in expressions.
        if (pinfo->procReturn.baseType == BaseType::UNKNOWN) {
            throw TypeError(1, "Semantic Analysis: procedure '" + node->funcName +
                                  "' used as expression but has no return type.");
        }
        node->type = pinfo->procReturn;
        return;
    }

    throw SymbolError(1, "Semantic Analysis: Unknown function/procedure '" +
                            node->funcName + "' in expression.");
}

void SemanticAnalysisVisitor::visit(IfNode* node) {
    // Evaluate and type-check condition first
    node->cond->accept(*this);
    if (node->cond->type.baseType != BaseType::BOOL) {
        throw TypeError(1, "Semantic Analysis: if condition must be boolean; got '" + toString(node->cond->type) + "'.");
    }
    if (node->thenBlock) {
        node->thenBlock->accept(*this);
    } else if (node->thenStat) {
        node->thenStat->accept(*this);
    }

    if (node->elseBlock) {
        node->elseBlock->accept(*this);
    } else if (node->elseStat) {
        node->elseStat->accept(*this);
    }
}

void SemanticAnalysisVisitor::visit(BlockNode* node) {
    // New lexical scope; inherit loop/return context
    enterScopeFor(node, current_->isInLoop(), current_->getReturnType());
    for (const auto& d : node->decs) {
        d->accept(*this);
    }
    // After processing declarations, prevent further declarations in this block
    current_->disableDeclarations();
    for (const auto& s : node->stats) {
        s->accept(*this);
    }
    exitScope();
}

void SemanticAnalysisVisitor::visit(LoopNode* node) {
    // Optional condition must be boolean
    if (node->cond) {
        node->cond->accept(*this);
        if (node->cond->type.baseType != BaseType::BOOL) {
            throw TypeError(1, "Semantic Analysis: loop condition must be boolean; got '" + toString(node->cond->type) + "'.");
        }
    }
    // Enter loop scope so 'break'/'continue' are legal
    enterScopeFor(node, true, current_->getReturnType());
    if (node->body) {
        node->body->accept(*this);
    }
    exitScope();
}

void SemanticAnalysisVisitor::visit(ParenExpr* node) {
    node->expr->accept(*this);
    node->type = node->expr->type;
}

/* TODO pt2
    - check element-wise types for arrays, vectors, matrices
*/
void SemanticAnalysisVisitor::visit(UnaryExpr* node) {
    node->operand->accept(*this); // eval operand type

    if (node->operand->type.baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("Semantic Analysis error: Applying operator '" + node->op + "' to operand with type UNKNOWN.");
    }

    std::string op = node->op;
    if (op == "-" || op == "+") {
        // Not permitted: bool, character, tuple, struct, string
        // permitted: int, real, arrays(int, real), vector(int, real), matrix(int, real)
        const BaseType illegalTypes[] = {BaseType::BOOL, BaseType::CHARACTER, BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING};

        if (std::find(std::begin(illegalTypes), std::end(illegalTypes), node->operand->type.baseType) != std::end(illegalTypes)) {
            throwOperandError(op, {node->operand->type}, "");
        }

    } else if (op == "not") {
        // Not permitted: character, int, real, tuple, struct, string
        // permitted: bool, arrays(bool), vector(bool), matrices(bool)
        const BaseType illegalTypes[] = {BaseType::CHARACTER, BaseType::INTEGER, BaseType::REAL, BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING};

        if (std::find(std::begin(illegalTypes), std::end(illegalTypes), node->operand->type.baseType) != std::end(illegalTypes)) {
            throwOperandError(op, {node->operand->type}, "");
        }

    } else {
        throw std::runtime_error("Semantic Analysis error: Unknown unary operator '" + node->op + "'.");
    }

    if (op == "not") {
        node->type = BaseType::BOOL;
    } else {
        node->type = node->operand->type; 
    }    
}

/* TODO pt2
    - handle array/vector/matrix element-wise type + len checking
    - handle int/real -> array/vector/matrix promotion
*/
void SemanticAnalysisVisitor::visit(ExpExpr* node) {
    node->left->accept(*this);
    node->right->accept(*this);

    if (node->op != "^") {
        throw std::runtime_error("Semantic Analysis error: unexpected operator in exponentiation '" + node->op + "'.");
    }

    // only automatic type mixing: int -> real OR int -> array/
    // permitted: int, real, (array+vector+matrix(real, int)|same size)
    // not permitted: boolean, character, tuple, struct, string
    const BaseType illegalTypes[] = {BaseType::BOOL, BaseType::CHARACTER, BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING};

    const CompleteType& leftOperandType = node->left->type;
    const CompleteType& rightOperandType = node->right->type;

    // Ensure both operands legal
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), leftOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError("^", {leftOperandType}, "Illegal left operand");
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError("^", {rightOperandType}, "Illegal right operand");
    }

    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        throwOperandError("^", {leftOperandType, rightOperandType}, "No promotion possible between operands");
    }

    node->type = finalType;
}

/* TODO pt2
    - handle array/vector/matrix element-wise type + len checking. Note matrix mult. requires a special check
    - pt2 handle int/real -> array/vector promotion. ONLY promote to matrix if square.
*/
void SemanticAnalysisVisitor::visit(MultExpr* node) {
    node->left->accept(*this);
    node->right->accept(*this);

    const std::unordered_map<std::string, std::string> legalOperators = {
        {"*", "multiplication"},
        {"/", "division"},
        {"%", "remainder"} 
    };

    if (legalOperators.find(node->op) == legalOperators.end()) {
        throw std::runtime_error("Semantic Analysis error: unexpected operator in mult/div/rem node '" + node->op + "'.");
    }

    // only automatic type mixing: int -> real OR int -> array/
    // permitted: int, real, (array+vector+matrix(real, int)|same size)
    // not permitted: boolean, character, tuple, struct, string
    const BaseType illegalTypes[] = {BaseType::BOOL, BaseType::CHARACTER, BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING};

    const CompleteType& leftOperandType = node->left->type;
    const CompleteType& rightOperandType = node->right->type;

    // Ensure both operands legal
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), leftOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {leftOperandType}, "Illegal left operand");
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {rightOperandType}, "Illegal right operand");
    }

    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        throwOperandError(node->op, {leftOperandType, rightOperandType}, "No promotion possible between operands");
    }

    node->type = finalType;
}

/* TODO pt2
    - handle array/vector/matrix element-wise type + len checking
    - pt2 handle int/real -> array/vector/matrix promotion.
*/
void SemanticAnalysisVisitor::visit(AddExpr* node) {
    node->left->accept(*this);
    node->right->accept(*this);

    std::set<std::string> legalOperators = {"+", "-"};

    if (legalOperators.find(node->op) == legalOperators.end()) {
        throw std::runtime_error("Semantic Analysis error: unexpected operator in add/sub node '" + node->op + "'.");
    }

    // only automatic type mixing: int -> real OR int -> array/
    // permitted: int, real, (array+vector+matrix(real, int)|same size)
    // not permitted: boolean, character, tuple, struct, string
    const BaseType illegalTypes[] = {BaseType::BOOL, BaseType::CHARACTER, BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING};
    const CompleteType& leftOperandType = node->left->type;
    const CompleteType& rightOperandType = node->right->type;

    // Ensure both operands legal
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), leftOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {leftOperandType}, "Illegal left operand");
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {rightOperandType}, "Illegal right operand");
    }

    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        throwOperandError(node->op, {leftOperandType, rightOperandType}, "No promotion possible between operands");
    }

    node->type = finalType;
}

/* TODO pt2
    - handle array/vector/matrix element-wise type + len checking. These should return compositeType<boolean>
    - handle int/real -> array/vector/matrix promotion.
*/
void SemanticAnalysisVisitor::visit(CompExpr* node) {
    node->left->accept(*this);
    node->right->accept(*this);

    std::set<std::string> legalOperators = {">", "<", ">=", "<="};

    if (legalOperators.find(node->op) == legalOperators.end()) {
        throw std::runtime_error("Semantic Analysis error: unexpected operator in compExpr node '" + node->op + "'.");
    }

    // only automatic type mixing: int -> real OR int -> array/
    // permitted: int, real, (array+vector+matrix(real, int)|same size)
    // not permitted: boolean, character, tuples, structs, string
    const BaseType illegalTypes[] = {BaseType::BOOL, BaseType::CHARACTER, BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING};
    const CompleteType& leftOperandType = node->left->type;
    const CompleteType& rightOperandType = node->right->type;

    // Ensure both operands legal
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), leftOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {leftOperandType}, "Illegal left operand");
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {rightOperandType}, "Illegal right operand");
    }

    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        throwOperandError(node->op, {leftOperandType, rightOperandType}, "No promotion possible between operands");
    }

    if (finalType.baseType == BaseType::INTEGER || finalType.baseType == BaseType::REAL) {
        node->type = BaseType::BOOL;
    } else {
        node->type = finalType; 
    }
}


/* TODO pt2 handle element wise checking of bools for composite types */
void SemanticAnalysisVisitor::visit(NotExpr* node) {
    // Evaluate operand and ensure it's a valid type for logical not
    node->operand->accept(*this);
    const BaseType illegalTypes[] = {
        BaseType::CHARACTER, BaseType::INTEGER, BaseType::REAL,
        BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING
    };
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), node->operand->type.baseType) != std::end(illegalTypes)) {
        throwOperandError("not", {node->operand->type}, "");
    }

    // Propagate type, i.e. bools remain bools, array/vec/matrix remain array/vec/matrix
    node->type = node->operand->type; 
}

void SemanticAnalysisVisitor::visit(TrueNode* node) {
    node->type = BaseType::BOOL;
}

void SemanticAnalysisVisitor::visit(FalseNode* node) {
    node->type = BaseType::BOOL;
}

void SemanticAnalysisVisitor::visit(CharNode* node) {
    node->type = BaseType::CHARACTER;
}

void SemanticAnalysisVisitor::visit(IntNode* node) {
    node->type = BaseType::INTEGER;
}

void SemanticAnalysisVisitor::visit(RealNode* node) {
    node->type = BaseType::REAL;
}

void SemanticAnalysisVisitor::visit(StringNode* node) {
    node->type = BaseType::STRING;
}

void SemanticAnalysisVisitor::visit(IdNode* node) {
    VarInfo* varInfo = current_->resolveVar(node->id); // handles no-declr
    node->type = varInfo->type;
    node->binding = varInfo;
}

void SemanticAnalysisVisitor::visit(TupleLiteralNode* node) {
    CompleteType literalType = CompleteType(BaseType::TUPLE);
    literalType.subTypes.reserve(node->elements.size());

    if (node->elements.size() < 2) {
        throw LiteralError(1, "All tuples must have at least 2 elements, not " + std::to_string(node->elements.size()) + ".");
    }

    // FIXME confirm and handle case where tuple<vector<tuple...>>.
    for (auto& exprNode: node->elements) {
        exprNode->accept(*this);
        if (exprNode->type.baseType == BaseType::TUPLE) {throw LiteralError(1, "Cannot have nested tuples.");
        } else if (exprNode->type.baseType == BaseType::UNKNOWN) {throw std::runtime_error("Semantic Analysis: FATAL: Cannot use UNKNOWN type inside tuples.");}

        literalType.subTypes.push_back(exprNode->type);
    }

    node->type = literalType;
}

void SemanticAnalysisVisitor::visit(TupleAccessNode* node) {
    VarInfo* varInfo = current_->resolveVar(node->tupleName);
    
    if (!varInfo) {
        throw std::runtime_error("Semantic Analysis: FATAL: Variable '" + node->tupleName + "' not found in TupleAccessNode");
    }

    if (varInfo->type.baseType != BaseType::TUPLE) {
        throw std::runtime_error("Semantic Analysis: FATAL: Non-tuple type '" + toString(varInfo->type) + "' in TupleAccessNode");
    }

    if (node->index > varInfo->type.subTypes.size() || node->index == 0) {
        IndexError(("Index " + std::to_string(node->index) + " out of range for tuple of len " + std::to_string(varInfo->type.subTypes.size())).c_str());
        return; 
    }

    // Bind this access node to the underlying tuple variable so codegen
    // doesn't need to re-resolve names and can honour declaration order.
    node->binding = varInfo;
    node->type = varInfo->type.subTypes[node->index - 1];
}

void SemanticAnalysisVisitor::visit(TypeCastNode* node) {
    // Evaluate operand first
    node->expr->accept(*this);
    // Resolve target type: prefer explicit alias name when present; otherwise
    // use the concrete base type provided by the AST.
    CompleteType target(BaseType::UNKNOWN);
    if (!node->targetAliasName.empty()) {
        target = *current_->resolveAlias(node->targetAliasName);
    } else {
        CompleteType tname = node->targetType;
        if (tname.baseType == BaseType::BOOL ||
            tname.baseType == BaseType::CHARACTER ||
            tname.baseType == BaseType::INTEGER ||
            tname.baseType == BaseType::REAL ||
            tname.baseType == BaseType::TUPLE ||
            tname.baseType == BaseType::VECTOR ||
            tname.baseType == BaseType::ARRAY ||
            tname.baseType == BaseType::MATRIX ||
            tname.baseType == BaseType::STRUCT ||
            tname.baseType == BaseType::STRING) {
            target = tname;
        }
    }
    // Ensure cast is type-compatible using explicit cast rules
    if (!canCastType(node->expr->type, target)) {
        throw TypeError(1, std::string("Semantic Analysis: cannot cast from '") + toString(node->expr->type) + "' to '" + toString(target) + "'.");
    }
    node->type = target;
}

void SemanticAnalysisVisitor::visit(TupleTypeCastNode* node) {
    // Evaluate the expression being cast
    node->expr->accept(*this);
    
    // Set the result type to the target tuple type
    node->type = node->targetTupleType;
    
    // Validate that the cast is legal
    if (!canCastType(node->expr->type, node->targetTupleType)) {
        throw TypeError(1, std::string("Semantic Analysis: cannot cast from '") + toString(node->expr->type) + "' to '" + toString(node->targetTupleType) + "'.");
    }
}


/* TODO pt2
    - handle array/vector/matrix + tuple + element-wise type + len checking. Note that this operator yields true iff all elements of array/vector/matrix type are equal.
    - handle int/real -> array/vector/matrix promotion.
    - handle error throw when struct types mismatch
*/
void SemanticAnalysisVisitor::visit(EqExpr* node) {
   node->left->accept(*this);
    node->right->accept(*this);

    std::set<std::string> legalOperators = {"==", "!="};

    if (legalOperators.find(node->op) == legalOperators.end()) {
        throw std::runtime_error("Semantic Analysis error: unexpected operator in eqExpr node '" + node->op + "'.");
    }

    // only automatic type mixing: int -> real OR int -> array/
    // permitted: boolean,character, int, real, tuple, struct, (array+vector+matrix(real, int)|same size), string
    // not permitted: nothing
    const BaseType illegalTypes[] = {BaseType::UNKNOWN};
    const CompleteType& leftOperandType = node->left->type;
    const CompleteType& rightOperandType = node->right->type;

    // Ensure both operands legal
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), leftOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {leftOperandType}, "Illegal left operand");
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {rightOperandType}, "Illegal right operand");
    }

    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        throwOperandError(node->op, {leftOperandType, rightOperandType}, "No promotion possible between operands");
    }

    node->type = BaseType::BOOL;
}

void SemanticAnalysisVisitor::visit(AndExpr* node) {
    node->left->accept(*this);
    node->right->accept(*this);

    if (node->op != "and") {
        throw std::runtime_error("Semantic Analysis error: unexpected operator in 'and' node '" + node->op + "'.");
    }

    const CompleteType& leftOperandType = node->left->type;
    const CompleteType& rightOperandType = node->right->type;
    if (leftOperandType.baseType != BaseType::BOOL || rightOperandType.baseType != BaseType::BOOL) {
        throwOperandError(node->op, {leftOperandType, rightOperandType}, "Illegal operand(s) in 'and' expr.");
    }

    node->type = BaseType::BOOL;
}

void SemanticAnalysisVisitor::visit(OrExpr* node) {
    node->left->accept(*this);
    node->right->accept(*this);

    std::set<std::string> legalOperators = {"or", "xor"};
    if (legalOperators.find(node->op) == std::end(legalOperators)) {
        throw std::runtime_error("Semantic Analysis error: unexpected operator in or/xor node '" + node->op + "'.");
    }

    const CompleteType& leftOperandType = node->left->type;
    const CompleteType& rightOperandType = node->right->type;
    if (leftOperandType.baseType != BaseType::BOOL || rightOperandType.baseType != BaseType::BOOL) {
        throwOperandError(node->op, {leftOperandType, rightOperandType}, "Illegal operand(s) in or/xor expr.");
    }

    node->type = BaseType::BOOL;
}

void SemanticAnalysisVisitor::throwOperandError(const std::string op, const std::vector<CompleteType>& operands, std::string additionalInfo) {
    std::stringstream ss;
    ss << "Semantic Analysis error: Applying operator '" << op << "' to operand";

    if (operands.size() > 1) ss << "s";
    ss << ": ";

    for (size_t i = 0; i < operands.size(); ++i) {
        ss << "'" << toString(operands[i]) << "'";
        if (i < operands.size() - 1) ss << ", ";
    }

    if (!additionalInfo.empty()) {
        ss << "\n" + additionalInfo;
    }

    throw std::runtime_error(ss.str());
}

// TODO: add source line/column once AST carries location info
/*
If empty string provided, prints non-variable promotion error msg
*/
void SemanticAnalysisVisitor::handleAssignError(const std::string varName, const CompleteType &varType, const CompleteType &exprType) {
    // Normalise any alias-based types before checking compatibility.
    CompleteType resolvedVarType = resolveUnresolvedType(current_, varType);
    CompleteType resolvedExprType = resolveUnresolvedType(current_, exprType);

    // Encapsulate the type compatibility check here
    if (promote(resolvedExprType, resolvedVarType) != resolvedVarType) {
        if (varName != "") {
            TypeError err(
                1,
                std::string("Semantic Analysis: Cannot assign type '") + toString(resolvedExprType) +
                "' to variable '" + varName + "' of type '" + toString(resolvedVarType) + "'."
            );
            throw err;
        } else {
            TypeError err(
                1,
                std::string("Semantic Analysis: Cannot assign type '") + toString(resolvedExprType) +
                "' to expected type '" + toString(resolvedVarType) + "'."
            );
            throw err;
        }

    }
}

void SemanticAnalysisVisitor::enterScopeFor(const ASTNode* ownerCtx, const bool inLoop, const CompleteType* returnType) {
    // Init root
    if (current_ == nullptr) {
        root_ = std::make_unique<Scope>(nullptr, inLoop, returnType);
        current_ = root_.get();
    }
    Scope* child = current_->createChild(inLoop, returnType);
    scopeByCtx_[ownerCtx] = child;
    current_ = child;
}

void SemanticAnalysisVisitor::exitScope() {
  if (current_ && current_->parent()) {
    current_ = current_->parent();
  }
}

/* Return check: a return anywhere ends the path OR if/else must both return */
bool SemanticAnalysisVisitor::guaranteesReturn(const BlockNode* block) const {
    for (const auto& stat : block->stats) {
        if (std::dynamic_pointer_cast<ReturnStatNode>(stat)) {
            return true;
        }
        if (auto ifNode = std::dynamic_pointer_cast<IfNode>(stat)) {
            bool thenRet = ifNode->thenBlock ? guaranteesReturn(ifNode->thenBlock.get()) : false;
            bool elseRet = ifNode->elseBlock ? guaranteesReturn(ifNode->elseBlock.get()) : false;
            if (thenRet && elseRet) {
                return true;
            }
        }
        // LoopNode does not guarantee return
    }
    return false;
}

const std::unordered_map<const ASTNode*, Scope*>& SemanticAnalysisVisitor::getScopeMap() const {
    return scopeByCtx_;
}
