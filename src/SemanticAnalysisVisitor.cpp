#include "SemanticAnalysisVisitor.h"

void SemanticAnalysisVisitor::visit(FileNode* node) {
    // Init and enter global scope
    // TODO: handle type aliases here
    // note: can shadow other symbol names
    scopeByCtx_.clear();
    current_ = nullptr;
    enterScopeFor(node);

    for (const auto& line: node->stats) {
        line->accept(*this);
    }

    exitScope();
}

void SemanticAnalysisVisitor::visit(BlockNode* node) {
    // Init and enter block scope
    enterScopeFor(node);

    for (const auto& line: node->stats) {
        line->accept(*this);
    }

    exitScope();
}

void SemanticAnalysisVisitor::visit(TypedDecNode* node) {
    node->init->accept(*this);
    node->type_alias->accept(*this);

    bool isConst = true;
    if (node->qualifier == "var") {
        isConst = false;
    } else if (node->qualifier == "const") {
    } else {
        throw std::runtime_error("Semantic Analysis: Invalid qualifier provided for typed declaration '" + node->qualifier + "'.");
    }

    CompleteType& varType = node->type_alias->type;

    // Ensure not already declared in scope
    if (!current_->declareVar(node->name, varType, isConst)) {
        throw std::runtime_error("Semantic Analysis: redeclaration of var '" + node->name + "'.");
    }

    // Ensure init expr type matches with var type (if provided)
    if (node->init != nullptr && promote(node->init->type, varType) != varType) {
        throwAssignError(node->name, varType, node->init->type);
    }

    node->type = varType;
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

    node->type = varType;
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

void throwAssignError(const std::string varName, const CompleteType &varType, const CompleteType &exprType) {
    throw std::runtime_error("Cannot assign type '" + toString(exprType) + "' to variable '" + varName + "' of type '" + toString(varType) + "'.");
}

void SemanticAnalysisVisitor::enterScopeFor(const ASTNode* ownerCtx) {
    // Init root
    if (current_ == nullptr) {
    root_ = std::make_unique<Scope>(nullptr);
    current_ = root_.get();
  }
  Scope* child = current_->createChild();
  scopeByCtx_[ownerCtx] = child;
  current_ = child;
}

void SemanticAnalysisVisitor::exitScope() {
  if (current_ && current_->parent()) {
    current_ = current_->parent();
  }
}