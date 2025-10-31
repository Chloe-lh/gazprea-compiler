#include "SemanticAnalysisVisitor.h"
#include "CompileTimeExceptions.h"
#include <unordered_set>

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
}

/* TODO insert line number for error
*/
void SemanticAnalysisVisitor::visit(FuncStatNode* node) {
    std::vector<VarInfo> params;
    params.reserve(node->parameters.size());
    std::unordered_set<std::string> paramNames;
    for (const auto& [paramType, paramName] : node->parameters) {
        if (paramName.empty()) {
            // Should not happen
            throw std::runtime_error("Semantic Analysis: FATAL: parameter name required in function definition '" + node->name + "'.");
        }
        if (!paramNames.insert(paramName).second) {
            throw SymbolError(1, "Semantic Analysis: duplicate parameter name '" + paramName + "' in function '" + node->name + "'.");
        }
        params.push_back(VarInfo{paramName, paramType, true});
    }

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
    for (const auto& v : params) {
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
    if (node->init) {
        node->init->accept(*this);
    }

    bool isConst = true;
    if (node->qualifier == "var") {
        isConst = false;
    } else if (node->qualifier == "const") {
    } else {
        throw std::runtime_error("Semantic Analysis: Invalid qualifier provided for typed declaration '" + node->qualifier + "'.");
    }

    // Declared type is carried as a CompleteType on the alias node
    CompleteType& varType = node->type_alias->type;

    // Ensure not already declared in scope
    current_->declareVar(node->name, varType, isConst);

    // Ensure init expr type matches with var type (if provided)
    if (node->init != nullptr) {
        handleAssignError(node->name, varType, node->init->type);
    }

    node->type = varType;
}

void SemanticAnalysisVisitor::visit(FuncPrototypeNode* node) {
    // Convert parameter list to VarInfo (names may be empty for prototypes)
    std::vector<VarInfo> params;
    params.reserve(node->parameters.size());
    for (const auto& [paramType, paramName] : node->parameters) {
        // Use paramName as-is (may be empty)
        params.push_back(VarInfo{paramName, paramType, true});
    }

    // Declare the function signature in the current (global) scope
    // Function prototypes may omit param names so we check that
    try {
        current_->declareFunc(node->name, params, node->returnType);
    } catch (...) {
        FuncInfo* existing = current_->resolveFunc(node->name, params);
        if (existing->funcReturn != node->returnType) {
            throw std::runtime_error("Semantic Analysis: conflicting return type for function prototype '" + node->name + "'.");
        }
    }
}

/* TODO add line numbers */
void SemanticAnalysisVisitor::visit(FuncBlockNode* node) {
    // Build parameter VarInfos (const by default)
    std::vector<VarInfo> params;
    params.reserve(node->parameters.size());
    std::unordered_set<std::string> paramNames;
    for (const auto& [paramType, paramName] : node->parameters) {
        if (paramName.empty()) { 
            // should not happen
            throw std::runtime_error("Semantic Analysis: FATAL: parameter name required in function definition '" + node->name + "'.");
        }
        if (!paramNames.insert(paramName).second) {
            throw SymbolError(1, "Semantic Analysis: duplicate parameter name '" + paramName + "' in function '" + node->name + "'.");
        }
        params.push_back(VarInfo{paramName, paramType, true});
    }

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
    for (const auto& v : params) {
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

void SemanticAnalysisVisitor::visit(InferredDecNode* node) {
    node->init->accept(*this);

    bool isConst = true;
    if (node->qualifier == "var") {
        isConst = false;
    } else if (node->qualifier == "const") {
    } else {
        throw std::runtime_error("Semantic Analysis: Invalid qualifier provided for type inference '" + node->qualifier + "'.");
    }

    CompleteType& varType = node->init->type; // no need to check promotability


    // Ensure not already declared in scope
    current_->declareVar(node->name, varType, isConst);

    node->type = varType;
}

void SemanticAnalysisVisitor::visit(TupleTypedDecNode* node) {
    if (node->init) {
        node->init->accept(*this);
    }
    // For tuple-typed declarations, the declared type is already present
    // on the declaration node as a CompleteType
    CompleteType& varType = node->type;

    // Ensure not already declared in scope
    current_->declareVar(node->name, varType, false);

    // Ensure init expr type matches with var type (if provided)
    if (node->init != nullptr) {
        handleAssignError(node->name, varType, node->init->type);
    }

    node->type = varType;
}

void SemanticAnalysisVisitor::visit(TypeAliasDecNode* node) {
    current_->declareAlias(node->alias, node->type);

    // assume node has been initialized with correct type
}

void SemanticAnalysisVisitor::visit(TypeAliasNode *node) {
    if (node->aliasName != "") {
        node->type = *current_->resolveAlias(node->aliasName);
    }

    // if no alias, assume node already initialized with correct type
}

void SemanticAnalysisVisitor::visit(TupleTypeAliasNode *node) {
    if (node->aliasName != "") {
        node->type = *current_->resolveAlias(node->aliasName);
    }

    // if no alias, assume node already initialized with correct type
}

void SemanticAnalysisVisitor::visit(AssignStatNode* node) {
    node->expr->accept(*this);

    // handles if undeclared
    const VarInfo* varInfo = current_->resolveVar(node->name);
    
    if (varInfo->isConst) {
        throw AssignError(1, "Semantic Analysis: cannot assign to const variable '" + node->name + "'."); // TODO add line num
    }

    handleAssignError(node->name, varInfo->type, node->expr->type);

    node->type = varInfo->type;
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
    if (!current_->isInFunction()) {
        throw StatementError(1, "Cannot use 'return' outside of function."); // TODO add line num
    }

    node->expr->accept(*this);

    // Check if matches return type
    handleAssignError("", *current_->getReturnType(), node->expr->type);
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
    // Encapsulate the type compatibility check here
    if (promote(exprType, varType) != varType) {
        if (varName != "") {
            TypeError err(
                1,
                std::string("Semantic Analysis: Cannot assign type '") + toString(exprType) +
                "' to variable '" + varName + "' of type '" + toString(varType) + "'."
            );
            throw err;
        } else {
            TypeError err(
                1,
                std::string("Semantic Analysis: Cannot assign type '") + toString(exprType) +
                "' to expected type '" + toString(varType) + "'."
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
