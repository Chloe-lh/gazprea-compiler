#include "SemanticAnalysisVisitor.h"

void SemanticAnalysisVisitor::visit(FileNode* node) {
    // Init and enter global scope
    scopeByCtx_.clear();
    current_ = nullptr;
    enterScopeFor(node);

    for (std::shared_ptr<ASTNode> statNode: node->statements) {
        statNode->accept(*this);
    }
}

void SemanticAnalysisVisitor::visit(CondNode* node) {
    // Check cond type == int
    node->ifCond->accept(*this);
    if (node->ifCond->type != BaseType::INTEGER) {
        throw std::runtime_error("TypeError: Non-numeric type '" + toString(node->ifCond->type) + "' cannot be used in conditional.");
    }

    // Eval body
    enterScopeFor(node);
    for (std::shared_ptr<ASTNode> statNode: node->body) {
        statNode->accept(*this);
    }
    exitScope();
}

void SemanticAnalysisVisitor::visit(LoopNode* node) {
    // Check cond type == int
    node->loopCond->accept(*this);
    if (node->loopCond->type != BaseType::INTEGER) {
        throw std::runtime_error("TypeError: Non-numeric type '" + toString(node->loopCond->type) + "' cannot be used in conditional.");
    }

    // Eval body
    enterScopeFor(node);
    for (std::shared_ptr<ASTNode> statNode: node->body) {
        statNode->accept(*this);
    }
    exitScope();
}

void SemanticAnalysisVisitor::visit(IntDecNode* node) {
    node->value->accept(*this);

    // Ensure type coersion is possible
    if (node->value->type != BaseType::INTEGER) {
        throw std::runtime_error("TypeError: Non-numeric type '" + toString(node->value->type) + "' cannot be declared to identifier of type int.");
    }

    current_->declare(node->id->name, BaseType::INTEGER);
    node->id->accept(*this);
}

void SemanticAnalysisVisitor::visit(VectorDecNode* node) {
    node->vectorValue->accept(*this);
    current_->declare(node->id->name, BaseType::VECTOR);
    node->id->accept(*this);

    // Ensure type coersion is possible
    if (node->vectorValue->type != BaseType::VECTOR) {
        throw std::runtime_error("TypeError: Non-vec type '" + toString(node->vectorValue->type) + "' cannot be declared to identifier of type vector.");
    }
}

void SemanticAnalysisVisitor::visit(AssignNode* node) {
    node->value->accept(*this);
    node->id->accept(*this);

    // Ensure type coersion is possible
    if (promote(node->value->type, node->id->type) != node->id->type) {
        throw std::runtime_error("TypeError: Value with type '" + toString(node->value->type) + "' cannot be assigned to identifier '" + node->id->name + "' with type '" + toString(node->id->type) + "'.");
    }    
}

void SemanticAnalysisVisitor::visit(IntNode *node) {
    node->type = BaseType::INTEGER;
}

void SemanticAnalysisVisitor::visit(IdNode *node) {
    // Ensure var exists
    SymbolInfo *idInfo = current_->resolve(node->name);
    if (idInfo == nullptr) {
        throw std::runtime_error("ReferenceError: Identifier '" + node->name + "' assigned to before declaration.");
    }

    // Resolve var type
    node->type = idInfo->type;
}

void SemanticAnalysisVisitor::visit(BinaryOpNode *node) {
    node->left->accept(*this);
    node->right->accept(*this);

    if (node->left->type == BaseType::UNKNOWN) {
        throw std::runtime_error("TypeError: Invalid left operand type '" + toString(node->left->type) + "' for operator '" + node->op + "'");
    }

    if (node->right->type == BaseType::UNKNOWN) {
        throw std::runtime_error("TypeError: Invalid right operand type '" + toString(node->right->type) + "' for operator '" + node->op + "'");
    }

    // If vectors are involved then promote to vector
    node->type = BaseType::INTEGER;
    if (node->left->type == BaseType::VECTOR || node->right->type == BaseType::VECTOR) {
        node->type = BaseType::VECTOR;
    }
}

void SemanticAnalysisVisitor::visit(RangeNode *node) {
    node->start->accept(*this);
    node->end->accept(*this);

    if (node->start->type != BaseType::INTEGER) {
        throw std::runtime_error("TypeError: Invalid left operand type '" + toString(node->start->type) + "' for range operator.");
    }

    if (node->end->type != BaseType::INTEGER) {
        throw std::runtime_error("TypeError: Invalid right operand type '" + toString(node->end->type) + "' for range operator.");
    }

    node->type = BaseType::VECTOR;
}

void SemanticAnalysisVisitor::visit(IndexNode* node) {
    node->array->accept(*this);
    node->index->accept(*this);

    if (node->array->type != BaseType::VECTOR) {
        throw std::runtime_error("TypeError: Non-vector type '" + toString(node->array->type) + "' used with index operator.");
    }

    if (node->index->type != BaseType::INTEGER) {
        throw std::runtime_error("TypeError: Non-int index type '" + toString(node->index->type) + "' used as index.");
    }

    node->type = BaseType::INTEGER;
}

void SemanticAnalysisVisitor::visit(GeneratorNode *node) {
    // "mini" scope to handle declr of domain variable
    enterScopeFor(node);
    current_->declare(node->id->name, BaseType::INTEGER);
    node->id->accept(*this);
    node->domain->accept(*this);
    node->body->accept(*this);

    if (node->domain->type != BaseType::VECTOR) {
        throw std::runtime_error("TypeError: Non-vec type '" + toString(node->domain->type) + "' used as domain in generator.");
    }

    if (node->body->type != BaseType::INTEGER) {
        throw std::runtime_error("TypeError: Non-int type '" + toString(node->body->type) + "' used as expression in generator.");
    }
    exitScope();

    node->type = BaseType::VECTOR;
}

void SemanticAnalysisVisitor::visit(FilterNode *node) {
    // "mini" scope to handle declr of domain variable
    enterScopeFor(node);
    current_->declare(node->id->name, BaseType::INTEGER);
    node->id->accept(*this);
    node->domain->accept(*this);
    node->predicate->accept(*this);

    if (node->domain->type != BaseType::VECTOR) {
        throw std::runtime_error("TypeError: Non-vec type '" + toString(node->domain->type) + "' used as domain in filter.");
    }

    if (node->predicate->type != BaseType::INTEGER) {
        throw std::runtime_error("TypeError: Non-int type '" + toString(node->predicate->type) + "' used as predicate in filter.");
    }
    exitScope();

    node->type = BaseType::VECTOR;
}

void SemanticAnalysisVisitor::visit(PrintNode *node) {
    node->printExpr->accept(*this);

    if (node->printExpr->type == BaseType::UNKNOWN) {
        throw std::runtime_error("TypeError: Invalid operand type '" + toString(node->printExpr->type) + "' for print.");
    }
}

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

        // TODO pt2: check element-wise types for arrays, vectors, matrices

    } else if (op == "not") {
        // Not permitted: character, int, real, tuple, struct, string
        // permitted: bool, arrays(bool), vector(bool), matrices(bool)
        const BaseType illegalTypes[] = {BaseType::CHARACTER, BaseType::INTEGER, BaseType::REAL, BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING};

        if (std::find(std::begin(illegalTypes), std::end(illegalTypes), node->operand->type.baseType) != std::end(illegalTypes)) {
            throwOperandError(op, {node->operand->type}, "");
        }

        // TODO pt2: check element-wise types for arrays, vectors, matrices
    } else {
        throw std::runtime_error("Semantic Analysis error: Unknown unary operator '" + node->op + "'.");
    }

    if (op == "not") {
        node->type = BaseType::BOOL;
    } else {
        node->type = node->operand->type; 
    }    
}

void SemanticAnalysisVisitor::visit(ExpExpr* node) {
    node->left->accept(*this);
    node->right->accept(*this);

    if (node->op != "^") {
        throw std::runtime_error("Semantic Analysis error: unexpected operator in exponentiation '" + node->op + "'.");
    }

    // only automatic type mixing: int -> real OR int -> array/
    // permitted: int, real, (array+vector+matrix(real, int)|same size)
    // not permitted: boolean, character, tuple, struct, string
    // TODO: pt2 handle array/vector/matrix element-wise type + len checking
    // TODO: pt2 handle int/real -> array/vector/matrix promotion
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
    // TODO: pt2 handle array/vector/matrix element-wise type + len checking. Note matrix mult. requires a special check
    // TODO: pt2 handle int/real -> array/vector promotion. ONLY promote to matrix if square.
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
    // TODO: pt2 handle array/vector/matrix element-wise type + len checking
    // TODO: pt2 handle int/real -> array/vector/matrix promotion.
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
    /* TODO pt2
        - handle array/vector/matrix + tuple + element-wise type + len checking. Note that this operator yields true iff all elements of array/vector/matrix type are equal.
        - handle int/real -> array/vector/matrix promotion.
        - handle error throw when struct types mismatch
    */
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

    // Handle element-wise tuple checks
    if (finalType.baseType == BaseType::TUPLE) {
        auto* leftTuple = dynamic_cast<TupleLiteralNode*>(node->left.get());
        auto* rightTuple = dynamic_cast<TupleLiteralNode*>(node->right.get());

        if (!(leftTuple && rightTuple)) {
            throwOperandError(node->op, {BaseType::TUPLE, BaseType::TUPLE}, "Failed to narrow Expr cast to Tuple literals");
        }
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