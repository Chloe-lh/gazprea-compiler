#include "SemanticAnalysisVisitor.h"
#include "AST.h"
#include "CompileTimeExceptions.h"
#include "Types.h"
#include <memory>
#include <optional>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <set>


// Resolve a CompleteType whose baseType is UNRESOLVED using the current scope's
// type information:
//   1. global type aliases (TYPEALIAS)
//   2. scoped struct types (struct declarations)
// Throws if UNRESOLVED and no alias name provided or no matching type found.
CompleteType SemanticAnalysisVisitor::resolveUnresolvedType(Scope *scope, const CompleteType &t, int line) {
    if (!scope) {
        return t;
    }

    // Resolve aliases iteratively until we reach a concrete type.
    // This handles cases where an alias might point to another alias.
    CompleteType result = t;
    while (result.baseType == BaseType::UNRESOLVED) {
        if (result.aliasName.empty()) {
            throw std::runtime_error(
                "Semantic Analysis: encountered UNRESOLVED type with no alias name.");
        }

        CompleteType *resolved = nullptr;

        // First try global/basic aliases.
        try {
            resolved = scope->resolveAlias(result.aliasName, line);
        } catch (const CompileTimeException &) {
            resolved = nullptr;
        }

        // Then try scoped struct types.
        if (!resolved) {
            try {
                resolved = scope->resolveStructType(result.aliasName, line);
            } catch (const CompileTimeException &) {
                resolved = nullptr;
            }
        }

        if (!resolved) {
            throw SymbolError(
                line, "Semantic Analysis: Type alias or struct type '" +
                          result.aliasName + "' not defined.");
        }

        result = *resolved;
    }

    // Recursively normalise any composite subtypes as well.
    if (!result.subTypes.empty()) {
        for (auto &sub : result.subTypes) {
            sub = resolveUnresolvedType(scope, sub, line);
        }
    }

    return result;
}

Scope* SemanticAnalysisVisitor::getRootScope() {
    return this->root_.get();
}

// Heuristic: Return the last line/declr in the block and add 1
static int computeMissingReturnLine(const BlockNode* body) {
    if (!body) return 1;
    int maxLine = body->line;
    for (const auto& d : body->decs) {
        if (d) maxLine = std::max(maxLine, d->line);
    }
    for (const auto& s : body->stats) {
        if (s) maxLine = std::max(maxLine, s->line);
    }
    return maxLine + 1;
}

void SemanticAnalysisVisitor::visit(FileNode* node) {
    // Init and enter global scope
    // TODO: handle type aliases here
    // note: can shadow other symbol names
    scopeByCtx_.clear();
    current_ = nullptr;
    enterScopeFor(node, false, nullptr);
    current_->setGlobalTrue();
    
    try {
        // making a dummy alias for the input stream
        current_->declareAlias("input_stream", CompleteType(BaseType::INTEGER), 0);
    } catch (...) { /* ignore redeclaration */ }

    // define 'stream_state' procedure
    try {
        std::vector<VarInfo> params;
        params.emplace_back("s", CompleteType(BaseType::INTEGER), true);
        
        CompleteType returnType(BaseType::INTEGER);
        
        // Declare in global scope
        current_->declareProc("stream_state", params, returnType, 0);
    } catch (...) { /* ignore redeclaration */ }


    for (const auto& stat: node->stats) {
        stat->accept(*this);
    }

    exitScope();

    // Ensure main exists
    if (!seenMain_) {
        throw MainError(node->line, "Semantic Analysis: procedure main() not defined.");
    }
}
void SemanticAnalysisVisitor::visit(ArrayStrideExpr *node) {
    // resolve the base array and the stride expression
    VarInfo* var = current_->resolveVar(node->id, node->line);
    if (!var) {
        throw SymbolError(node->line, "Semantic Analysis: unknown array '" + node->id + "'.");
    }
    CompleteType baseType = resolveUnresolvedType(current_, var->type, node->line);
    if (baseType.baseType != BaseType::ARRAY) {
        throw TypeError(node->line, "Semantic Analysis: stride operator applied to non-array type.");
    }
    // stride expression must be integer
    if (node->expr) {
        node->expr->accept(*this);
        if (node->expr->type.baseType != BaseType::INTEGER) {
            throw TypeError(node->line, "Semantic Analysis: stride expression must be integer.");
        }
    }
    // result of stride is an array of the same element type
    node->type = baseType;
}

void SemanticAnalysisVisitor::visit(ArraySliceExpr *node) {
    // Resolve array variable
    VarInfo* var = current_->resolveVar(node->id, node->line);
    if (!var) {
        throw SymbolError(node->line, "Semantic Analysis: unknown array '" + node->id + "'.");
    }
    CompleteType baseType = resolveUnresolvedType(current_, var->type, node->line);
    if (baseType.baseType != BaseType::ARRAY) {
        throw TypeError(node->line, "Semantic Analysis: slice operator applied to non-array type.");
    }
    // Validate range
    if (node->range) {
        node->range->accept(*this);
    }
    // validate range against compile-time array size if available
    // Note: We allow negative indices (they will be converted at runtime)
    VarInfo* varInfo = current_->resolveVar(node->id, node->line);
    if (node->range && varInfo && !varInfo->type.dims.empty() && varInfo->type.dims[0] >= 0) {
        auto sz = varInfo->type.dims[0];
        // If start/end are integer literals, check bounds (but allow negative indices)
        if (node->range->start) {
            if (auto in = std::dynamic_pointer_cast<IntNode>(node->range->start)) {
                // Only check bounds if index is positive and out of range
                // Negative indices are allowed and will be converted at runtime
                if (in->value > 0 && in->value > sz) {
                    throw SizeError(node->line, "Index " + std::to_string(in->value) + " out of range for array of len " + std::to_string(sz));
                    return;
                }
            }
        }
        if (node->range->end) {
            if (auto in = std::dynamic_pointer_cast<IntNode>(node->range->end)) {
                // Only check bounds if index is positive and out of range
                // Negative indices are allowed and will be converted at runtime
                if (in->value > 0 && in->value > sz) {
                    throw SizeError(node->line, "Index " + std::to_string(in->value) + " out of range for array of len " + std::to_string(sz));
                    return;
                }
            }
        }
    }
    // slicing yields an array of the same element type, but with dynamic dimensions
    node->type = baseType;
    node->type.dims.clear();
    node->type.dims.push_back(-1); // Dynamic dimension
}

// Built-in functions with a single identifier argument (length/len, shape,
// reverse, format).
void SemanticAnalysisVisitor::visit(BuiltInFuncNode *node) {
    std::string fname = node->funcName;
    VarInfo* var = current_->resolveVar(node->id, node->line);
    if (!var) {
        throw SymbolError(node->line, "Semantic Analysis: unknown identifier '" + node->id + "' in builtin call.");
    }
    CompleteType argType = resolveUnresolvedType(current_, var->type, node->line);

    if (fname == "length") {
        if (argType.baseType == BaseType::STRING || argType.baseType == BaseType::ARRAY ||
            argType.baseType == BaseType::VECTOR || argType.baseType == BaseType::MATRIX) {
            node->type = CompleteType(BaseType::INTEGER);
            return;
        }
        throw TypeError(node->line, "Semantic Analysis: length() requires string/array/vector/matrix argument.");
    }

    if (fname == "shape") {
        if (argType.baseType == BaseType::ARRAY || argType.baseType == BaseType::VECTOR || argType.baseType == BaseType::MATRIX) {
            if (argType.dims.empty()) {
                throw TypeError(node->line, "Semantic Analysis: shape() requires a sized array/vector/matrix.");
            }
            CompleteType elem(BaseType::INTEGER);
            node->type = CompleteType(BaseType::ARRAY, elem, {static_cast<int>(argType.dims.size())});
            return;
        }
        throw TypeError(node->line, "Semantic Analysis: shape() requires array/vector/matrix argument.");
    }

    if (fname == "reverse") {
        if (argType.baseType == BaseType::STRING || argType.baseType == BaseType::ARRAY ||
            argType.baseType == BaseType::VECTOR) {
            node->type = argType;
            return;
        }
        throw TypeError(node->line, "Semantic Analysis: reverse() requires string/array/vector argument.");
    }

    if (fname == "format") {
        if (argType.baseType == BaseType::STRING) {
            node->type = argType;
            return;
        }
        throw TypeError(node->line, "Semantic Analysis: format() requires string argument.");
    }

    throw SymbolError(node->line, "Semantic Analysis: unknown builtin function '" + node->funcName + "'.");
}

void SemanticAnalysisVisitor::visit(ArrayAccessNode *node) {
    // Visit the index expression first to determine its type
    if (!node->indexExpr) {
        throw std::runtime_error("Semantic Analysis: ArrayAccessNode has no index expression.");
    }
    node->indexExpr->accept(*this);
    if (node->indexExpr->type.baseType != BaseType::INTEGER) {
        throw TypeError(node->line, "Semantic Analysis: array index must be an integer expression.");
    }

    // Resolve array variable
    VarInfo* var = current_->resolveVar(node->id, node->line);
    if (!var) {
        throw SymbolError(node->line, "Semantic Analysis: unknown array '" + node->id + "'.");
    }

    CompleteType baseType = resolveUnresolvedType(current_, var->type, node->line);
    if (baseType.baseType != BaseType::ARRAY && baseType.baseType != BaseType::VECTOR) {
        throw TypeError(node->line, "Semantic Analysis: index operator applied to non-array type.");
    }

    // Perform compile-time bounds check if possible (for integer literals)
    if (auto intLiteral = std::dynamic_pointer_cast<IntNode>(node->indexExpr)) {
        if (!var->type.dims.empty() && var->type.dims[0] >= 0) {
            int indexValue = intLiteral->value;
            int arraySize = var->type.dims[0];
            // Using 1-based indexing - index must be between 1 and arraySize (or negative for reverse indexing)
            if (indexValue == 0 || indexValue > arraySize || indexValue < -arraySize) {
                throw SizeError(node->line, "Index " + std::to_string(indexValue) + " out of range for array of length " +
                    std::to_string(arraySize));
                return;
            }
        }
    }
    
    // Handle 2D array access - check second index if present
    if (node->indexExpr2) {
        node->indexExpr2->accept(*this);
        if (node->indexExpr2->type.baseType != BaseType::INTEGER) {
            throw TypeError(node->line, "Semantic Analysis: second array index must be an integer expression.");
        }
        
        // Perform compile-time bounds check for second dimension if possible
        if (auto intLiteral2 = std::dynamic_pointer_cast<IntNode>(node->indexExpr2)) {
            if (var->type.dims.size() >= 2 && var->type.dims[1] >= 0) {
                int indexValue2 = intLiteral2->value;
                int arraySize2 = var->type.dims[1];
                if (indexValue2 == 0 || indexValue2 > arraySize2 || indexValue2 < -arraySize2) {
                    throw SizeError(node->line, "Second index " + std::to_string(indexValue2) + " out of range for dimension of length " +
                        std::to_string(arraySize2));
                    return;
                }
            }
        }
    }

    if (baseType.subTypes.size() != 1) {
        throw std::runtime_error("Semantic Analysis: ARRAY type must have exactly one element subtype.");
    }
    node->type = baseType.subTypes[0];

    // Attach variable for codegen
    node->binding = var;
}


void SemanticAnalysisVisitor::visit(ArrayTypedDecNode *node) {
    if (!current_) {
        throw std::runtime_error("SemanticAnalysisVisitor: current scope is null");
    }

    if (!current_->isDeclarationAllowed()) {
        throw DefinitionError(node->line,
            "Declarations must appear at the top of a block.");
    }

    // Resolve any aliases inside the declared type.
    CompleteType declaredType = resolveUnresolvedType(current_, node->type, node->line);
    // Permit ARRAY, VECTOR, and MATRIX declarations here; 2D arrays/matrices
    // share the same declaration node shape.
    if (declaredType.baseType != BaseType::ARRAY &&
        declaredType.baseType != BaseType::VECTOR &&
        declaredType.baseType != BaseType::MATRIX) {
        throw TypeError(node->line,
            "ArrayTypedDecNode cannot be used for type '" + toString(declaredType) + "', at declaration '" + node->name + "'.");
    }

    node->type = declaredType;

    // Qualifier checks: arrays default to mutable unless explicitly 'const'
    if (node->qualifier.empty()) node->qualifier = "var";
    bool isConst = (node->qualifier == "const");
    // Older qualifier usage: empty qualifier should still default to mutable
    if (!isConst && current_->isInGlobal()) {
        throw GlobalError(node->line, "'var' is not allowed in global scope.");
    }

    // Declare variable
    current_->declareVar(node->id, declaredType, isConst, node->line);
    VarInfo* declared = current_->resolveVar(node->id, node->line);
    if (!declared) {
        throw std::runtime_error("Failed to declare variable: " + node->id);
    }

    // Handle initializer
    if (node->init) {
        // std::cerr << "[DEBUG] Initializer present for " << node->id << "\n";
        node->init->accept(*this);
        CompleteType initType = resolveUnresolvedType(current_, node->init->type, node->line);
        // std::cerr << "[DEBUG] Initializer type: " << toString(initType) << "\n";

        // 1. Handle array literals as initializer
        if (auto lit = std::dynamic_pointer_cast<ArrayLiteralNode>(node->init)) {
            int64_t litSize = lit->list ? lit->list->list.size() : 0;
            // Empty literals carry UNKNOWN element types; seed them from the declared type
            if (litSize == 0 && !declaredType.subTypes.empty()) {
                CompleteType elemType = declaredType.subTypes[0];
                lit->type.subTypes = {elemType};
                initType.subTypes = {elemType};
                if (lit->type.dims.empty()) {
                    lit->type.dims = {0};
                } else if (lit->type.dims[0] < 0) {
                    lit->type.dims[0] = 0;
                }
            }
            // If dims not provided, infer from literal (1D or nested 2D)
            if (declaredType.dims.empty()) {
                bool isNested = lit->list && !lit->list->list.empty() &&
                                 std::dynamic_pointer_cast<ArrayLiteralNode>(lit->list->list[0]);
                if (isNested) {
                    int64_t lit2Size = 0;
                    if (auto innerLit = std::dynamic_pointer_cast<ArrayLiteralNode>(lit->list->list[0])) {
                        lit2Size = innerLit->list ? innerLit->list->list.size() : 0;
                    }
                    declaredType.dims = {static_cast<int>(litSize), static_cast<int>(lit2Size)};
                } else {
                    declaredType.dims = {static_cast<int>(litSize)};
                }
                declared->type.dims = declaredType.dims;
                node->type.dims = declaredType.dims;
            }
            //TODO add matrix dimensions
            // If `lhs` is array, then ensure its dims match + determine any inferred dimensions 
            if (declaredType.baseType == BaseType::ARRAY) {
                if (declaredType.dims[0] >= 0 && declaredType.dims[0] != litSize) {
                    throw TypeError(node->line,
                        "Array initializer length (" + std::to_string(litSize) + 
                        ") does not match declared size (" + std::to_string(declaredType.dims[0]) + ")");
                }

                // If `lhs` is array, update any inferred sizes to literal length
                if ( declaredType.dims[0] < 0) {
                    declaredType.dims[0] = static_cast<int>(litSize);
                    declared->type.dims = declaredType.dims;
                    node->type.dims = declaredType.dims;
                }
                //! currently not helping empty array case
                // // If the initializer is an empty array literal, it currently
                // // carries an EMPTY subtype and no element type information.
                // // Propagate the declared element subtype into the literal so
                // // subsequent promotion/type checks succeed (e.g. when the
                // // declared element type is numeric and the literal is empty).
                // if (litSize == 0) {
                //     lit->type = declaredType;
                // }
            }
            if(declaredType.baseType == BaseType::VECTOR){
                declaredType.dims[0] = static_cast<int>(litSize);
                declared->type.dims = declaredType.dims;
                node->type.dims = declaredType.dims;
            }
            
            // Handle 2D array/matrix dimensions
            if (declaredType.baseType == BaseType::MATRIX){
                // Ensure dims has two slots; fill missing with -1 (wildcard)
                if (declaredType.dims.size() < 2) {
                    declaredType.dims.resize(2, -1);
                }

                // integer[3][2] B = [[1, 2], [4, 5], [7, 8]];
                // Validate or infer first dimension (rows) (number of literals)
                if (declaredType.dims[0] >= 0) {
                    // Dimension explicitly declared - must match unless wildcard (*) == -1
                    if (declaredType.dims[0] != litSize) {
                        throw TypeError(node->line,
                            "Matrix initializer row count (" + std::to_string(litSize) + 
                            ") does not match declared size (" + std::to_string(declaredType.dims[0]) + ")");
                    }
                } else {
                    // Inferred dimension - set from literal size
                    declaredType.dims[0] = static_cast<int>(litSize);
                }

                int64_t lit2Size = 0;
                if (litSize > 0) {
                    auto innerLit = std::dynamic_pointer_cast<ArrayLiteralNode>(lit->list->list[0]);
                    if (!innerLit) {
                        throw TypeError(node->line, "Matrix initializer must use nested array literals for rows.");
                    }
                    lit2Size = innerLit->list ? static_cast<int64_t>(innerLit->list->list.size()) : 0;
                }
                // Validate or infer second dimension (columns = number of elements inside 1st array)
                if (declaredType.dims[1] >= 0) {
                    // Dimension explicitly declared - must match unless wildcard (*) == -1
                    if (declaredType.dims[1] != lit2Size) {
                        throw TypeError(node->line,
                            "Matrix initializer column count (" + std::to_string(lit2Size) + 
                            ") does not match declared size (" + std::to_string(declaredType.dims[1]) + ")");
                    }
                } else {
                    // Inferred dimension - set from literal
                    declaredType.dims[1] = static_cast<int>(lit2Size);
                }
                
                
                // Validate all rows have same column count
                for (size_t i = 1; i < lit->list->list.size(); ++i) {
                    if (auto innerLit = std::dynamic_pointer_cast<ArrayLiteralNode>(lit->list->list[i])) {
                        int64_t rowSize = innerLit->list ? innerLit->list->list.size() : 0;
                        if (rowSize != lit2Size) {
                            throw TypeError(node->line,
                                "Matrix initializer has inconsistent row lengths: row 0 has " + 
                                std::to_string(lit2Size) + " elements, row " + std::to_string(i) + 
                                " has " + std::to_string(rowSize) + " elements");
                        }
                    }
                }
                // Update declared variable with validated dimensions
                declared->type.dims = declaredType.dims;
                node->type.dims = declaredType.dims;
                
                // std::cerr << "[DEBUG] Before normalization: lit->type.dims.size()=" << lit->type.dims.size() << std::endl;
                // std::cerr << "[DEBUG] declaredType.baseType=" << toString(declaredType.baseType) << std::endl;
                // std::cerr << "[DEBUG] declaredType.dims.size()=" << declaredType.dims.size() << std::endl;
                
                // Normalize the initializer type to match declared type
                // (matrix and 2D array are semantically equivalent)
                // IMPORTANT: Only normalize the TOP-LEVEL literal, not nested row arrays
                if (declaredType.baseType == BaseType::MATRIX) {
                    // Update the literal node's type to match
                    if (lit->type.dims != declaredType.dims) {
                        throw SizeError(node->line, "Initializer dimensions do not match declared matrix dimensions");
                    }
                    
                    // Do NOT modify inner array types - they should remain as ARRAY with 1D
                } else if (declaredType.baseType == BaseType::ARRAY && declaredType.dims.size() == 2) {
                    // 2D array - ensure dims are set on both sides
                    initType.dims = declaredType.dims;
                    lit->type.dims = declaredType.dims;
                    // std::cerr << "[DEBUG] Set lit->type to ARRAY with dims.size()=" << lit->type.dims.size() << std::endl;
                }
            }
        // Handle array slice expression as initializer
        } else if (auto sliceExpr = std::dynamic_pointer_cast<ArraySliceExpr>(node->init)) {
            // ArraySliceExpr already sets its type to array with dynamic dimension
            // Just verify it's an array type
            if (initType.baseType != BaseType::ARRAY &&
                initType.baseType != BaseType::VECTOR) {
                throw std::runtime_error(
                    "SemanticAnalysis::ArrayTypedDecNode: ArraySliceExpr initializer must result in array/vector type for '" + node->id + "'.");
            }
        // Handle initializer as identifier or general expression
        } else {
            // Permit any expression initializer (including IdNode, DotExpr, etc.)
            // The type checking will be done below via promotion and handleAssignError
            if (initType.baseType == BaseType::UNKNOWN) {
                throw std::runtime_error(
                    "SemanticAnalysis::ArrayTypedDecNode: Expression initializer has UNKNOWN type for '" + node->id + "'.");
            }
        }
        handleGlobalErrors(node);

        // 2. For 2D arrays/matrices with validated literals, skip promotion and use validated type directly
        //! might need to facor in casting/promotion - but assign error was getting called on a matrix<T> matrix<T> and was failing
        CompleteType promoted;
        if (auto lit = std::dynamic_pointer_cast<ArrayLiteralNode>(node->init)) {
            if (declaredType.dims.size() == 2) {
                // Dimension validation already passed above; dimensions match
                promoted = declaredType;
            } else {
                // 1D array: standard promotion
                CompleteType resolvedInit = resolveUnresolvedType(current_, initType, node->line);
                promoted = promote(resolvedInit, declaredType);
            }
        } else {
            // Non-literal initializer: standard promotion
            CompleteType resolvedInit = resolveUnresolvedType(current_, initType, node->line);
            promoted = promote(resolvedInit, declaredType);
        }
        
        node->type = promoted;
        if(declaredType != promoted) handleAssignError(node->id, declaredType, promoted, node->line);
    } else if (declaredType.dims.empty()) {
        throw std::runtime_error("SemanticAnalysis::ArrayTypedDecNode: Empty dims for '" + node->id + "'.");
    } else if (declaredType.dims[0] == -1 && node->type.baseType == BaseType::ARRAY) {
        // Vectors are allowed to have dims=-1 because it represents dynamic sizing.
        throw StatementError(node->line, "Cannot have inferred type array without initializer");
    }

    // std::cerr << "[DEBUG] Finished ArrayTypedDecNode: " << node->id << "\n";
}

void SemanticAnalysisVisitor::visit(ExprListNode *node) {
    // std::cout << "[DEUBG] semantic analysis: visiting ExprListNode\n";
    for (auto &e : node->list) {
        if (e) e->accept(*this);
    }
}
void SemanticAnalysisVisitor::visit(DotExpr *node){ 
    // expressions must be resolved to a vector/matrix/array
    if(node->left) node->left->accept(*this);
    if(node->right) node->right->accept(*this);
    CompleteType leftType = resolveUnresolvedType(current_, node->left->type, node->line);
    CompleteType rightType = resolveUnresolvedType(current_, node->right->type, node->line);
    // left and right operands must be a vector/matrix/array
    if(!(leftType.baseType == BaseType::ARRAY || leftType.baseType == BaseType::VECTOR || leftType.baseType == BaseType::MATRIX)){
        throw TypeError(node->line, "Semantic Analysis: left expression in dot product must be a array/vector/matrix type");
    }
    if(!(node->right->type.baseType!= BaseType::ARRAY || node->right->type.baseType != BaseType::VECTOR || node->left->type.baseType!= BaseType::MATRIX)){
        throw TypeError(node->line, "Semantic Analysis: right expression in dot product must be a array/vector/matrix type");
    }
    auto is1D = [](const CompleteType &t){
        return t.baseType == BaseType::ARRAY || t.baseType == BaseType::VECTOR;
    };
    auto isMatrix=[](const CompleteType &t){
        return t.baseType == BaseType::MATRIX;
    };
    // extract element types
    CompleteType leftElem = leftType.subTypes.empty() ? CompleteType(BaseType::UNKNOWN) : leftType.subTypes[0];
    CompleteType rightElem = rightType.subTypes.empty() ? CompleteType(BaseType::UNKNOWN) : rightType.subTypes[0];
    // ensure element types are numeric
    CompleteType promoted = promote(leftElem, rightElem);
    if(promoted.baseType == BaseType::UNKNOWN) promoted = promote(rightElem, leftElem);
    if(promoted.baseType == BaseType::UNKNOWN) throw TypeError(node->line, "Semantic Analysis: element types must be numeric for dot product");

    auto getDim = [](const std::vector<int> &d, size_t i) -> int {
        if (i < d.size()) return d[i];
        return -1;
    };

    bool L1 = is1D(leftType);
    bool R1 = is1D(rightType);
    bool LM = isMatrix(leftType);
    bool RM = isMatrix(rightType);

    // check dimensions
    if(L1 && R1){ // vector ** vector = scalar
        int Llen = getDim(leftType.dims, 0);
        int Rlen = getDim(rightType.dims, 0);
        if(Llen < 0 || Rlen < 0) throw SizeError(node->line, "Semantic Analysis: invalid vector/array dimensions");
        if(Llen != Rlen) throw SizeError(node->line, "Semantic Analysis: vectors/arrays must have the same dimensions in order to calculate dot product");
        node->type = promoted;
        return;
    }else if(LM && RM){ // matrix ** matrix = matrix
         /*
        the number of columns of the first operand must equal the number of rows of the second operand, e.g. an 
        mxn matrix multiplied by an nxp matrix will produce an mxp matrix. If the dimensions are not correct a 
        SizeError should be raised.
        */
        int Lrow = getDim(leftType.dims, 0);
        int Lcol = getDim(leftType.dims, 1);
        int Rrow = getDim(rightType.dims, 0);
        int Rcol = getDim(rightType.dims, 1);
        if(Lrow<0||Lcol<0||Rrow<0||Rcol<0) throw SizeError(node->line, "Semantic Analysis: invalid matrix dimensions");
        if(Lcol != Rrow) throw SizeError(node->line, ("Semantic Analysis: invalid matrix dimensions for matrix multiplication - left columns (" + std::to_string(Lcol) + ") must equal right rows (" + std::to_string(Rrow) + ")"));
        
        // Ensure element type promotion succeeded
        if(promoted.baseType == BaseType::UNKNOWN) {
            throw TypeError(node->line, "Semantic Analysis: incompatible element types for matrix multiplication");
        }
        
        int outRows = Lrow;
        int outCols = Rcol;
        node->type = CompleteType(BaseType::MATRIX, promoted, {outRows, outCols});
        return;
    }
    // maybe vector and matrix
    node->type = CompleteType(BaseType::UNKNOWN);
}

void SemanticAnalysisVisitor::visit(ArrayLiteralNode *node) {
    // If no elements, represent as ARRAY with UNKNOWN element subtype so later passes can infer.
    if (!node->list || node->list->list.empty()) {
        node->type = CompleteType(BaseType::ARRAY, CompleteType(BaseType::UNKNOWN), {0});
        return;
    }

    // Visit children first
    for (auto &elem : node->list->list) {
        elem->accept(*this);
    }

    // Detect nested array literal (matrix-like)
    auto firstArray = std::dynamic_pointer_cast<ArrayLiteralNode>(node->list->list[0]);
    if (firstArray) {
        // All elements must be arrays
        int64_t cols = firstArray->list ? static_cast<int64_t>(firstArray->list->list.size()) : 0;
        CompleteType elementType = CompleteType(BaseType::UNKNOWN);

        for (size_t i = 0; i < node->list->list.size(); ++i) {
            auto inner = std::dynamic_pointer_cast<ArrayLiteralNode>(node->list->list[i]);
            if (!inner) {
                throw LiteralError(node->line, "Semantic Analysis: mixed scalar and array elements in array literal.");
            }

            CompleteType innerType = resolveUnresolvedType(current_, inner->type, node->line);
            if (innerType.baseType != BaseType::ARRAY || innerType.subTypes.empty()) {
                throw LiteralError(node->line, "Semantic Analysis: nested array literal missing element type.");
            }

            // Track element type across rows
            CompleteType innerElem = innerType.subTypes[0];
            if (i == 0) {
                elementType = innerElem;
            } else {
                CompleteType promoted = promote(innerElem, elementType);
                if (promoted.baseType == BaseType::UNKNOWN) promoted = promote(elementType, innerElem);
                if (promoted.baseType == BaseType::UNKNOWN) {
                    throw LiteralError(node->line, "Semantic Analysis: incompatible element types in nested array literal.");
                }
                elementType = promoted;
            }

            // Ensure consistent column counts
            int64_t rowCols = inner->list ? static_cast<int64_t>(inner->list->list.size()) : 0;
            if (rowCols != cols) {
                throw TypeError(node->line, "Matrix initializer has inconsistent row lengths.");
            }
        }

        int64_t rows = static_cast<int64_t>(node->list->list.size());
        node->type = CompleteType(BaseType::ARRAY, elementType, {static_cast<int>(rows), static_cast<int>(cols)});
        // std::cerr << "[DEBUG SEMANTIC] ArrayLiteralNode EXIT: 2D ARRAY with dims={" << rows << "," << cols << "}" << std::endl;
        return;
    }

    // 1D array literal
    CompleteType common = CompleteType(BaseType::UNKNOWN);
    for (size_t i = 0; i < node->list->list.size(); ++i) {
        auto &elem = node->list->list[i];
        CompleteType et = resolveUnresolvedType(current_, elem->type, node->line);
        if (i == 0) {
            common = et;
        } else {
            CompleteType promoted = promote(et, common);
            if (promoted.baseType == BaseType::UNKNOWN) promoted = promote(common, et);
            if (promoted.baseType == BaseType::UNKNOWN) {
                throw LiteralError(node->line, "Semantic Analysis: incompatible element types in array literal.");
            }
            common = promoted;
        }
    }

    node->type = CompleteType(BaseType::ARRAY, common, {static_cast<int>(node->list->list.size())});
    // std::cerr << "[DEBUG SEMANTIC] ArrayLiteralNode EXIT: 1D ARRAY with dims={" << node->list->list.size() << "}" << std::endl;
}

void SemanticAnalysisVisitor::visit(RangeExprNode *node) {
    // Validate start/end expressions when present; they must be integer
    if (node->start) {
        node->start->accept(*this);
        if (node->start->type.baseType != BaseType::INTEGER) {
            throw TypeError(node->line, "Semantic Analysis: range start must be integer.");
        }
    }
    if (node->end) {
        node->end->accept(*this);
        if (node->end->type.baseType != BaseType::INTEGER) {
            throw TypeError(node->line, "Semantic Analysis: range end must be integer.");
        }
    }
    node->type = CompleteType(BaseType::UNKNOWN);
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
            throw SymbolError(node->line, "Semantic Analysis: duplicate parameter name '" + v.identifier + "' in function '" + node->name + "'.");
        }
        param.type = resolveUnresolvedType(current_, param.type, node->line);
        params.push_back(param);
    }

    // Push resolved types onto params (handling type aliasing)
    for (size_t i = 0; i < node->parameters.size() && i < params.size(); ++i) {
        node->parameters[i].type = params[i].type;
    }

    // Resolve alias in return type, if any.
    node->returnType = resolveUnresolvedType(current_, node->returnType, node->line);

    try {
        std::cerr << "declareFunc: " << node->name << " line " << node->line << std::endl;
        current_->declareFunc(node->name, params, node->returnType, node->line);
    } catch (...) {
        // If already declared, ensure it resolves to the same signature
        FuncInfo* existing = current_->resolveFunc(node->name, params, node->line);
        if (existing->funcReturn != node->returnType) {
            throw std::runtime_error("Semantic Analysis: conflicting return type for function '" + node->name + "'.");
        }
    }

    // Enter function scope, bind parameters
    enterScopeFor(node, false, &node->returnType);
    current_->setInFunctionTrue(); 
    for (auto& v : params) {
        if (!v.isConst) { throw AssignError(node->line, "Non-const arg given as function parameter");}
        current_->declareVar(v.identifier, v.type, true, node->line);
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
        throw DefinitionError(node->line, "Semantic Analysis: Declarations must appear at the top of a block."); 
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
        isConst = false;
    } else if (node->qualifier == "const") {
    } else {
        throw std::runtime_error("Semantic Analysis: Invalid qualifier provided for typed declaration '" + node->qualifier + "'.");
    }

    handleGlobalErrors(node);

    // Declared type is carried as a CompleteType on the alias node
    // TypeAliasNode::visit() already resolves the alias and subtypes
    node->type_alias->accept(*this);
    CompleteType varType = resolveUnresolvedType(current_, node->type_alias->type, node->line);

    // Ensure not already declared in scope
    current_->declareVar(node->name, varType, isConst, node->line);

    // Ensure init expr type matches with var type (if provided)
    if (node->init != nullptr) {
        handleAssignError(node->name, varType, node->init->type, node->line);
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
        param.type = resolveUnresolvedType(current_, param.type, node->line);
        params.push_back(param);
    }

    // Propagate resolved parameter types back to the AST node
    for (size_t i = 0; i < node->parameters.size() && i < params.size(); ++i) {
        node->parameters[i].type = params[i].type;
    }

    node->returnType = resolveUnresolvedType(current_, node->returnType, node->line);

    // Declare the function signature in the current (global) scope
    // Function prototypes may omit param names so we check that
    try {
        current_->declareFunc(node->name, params, node->returnType, node->line);
    } catch (...) {
        FuncInfo* existing = current_->resolveFunc(node->name, params, node->line);
        if (existing->funcReturn != node->returnType) {
            throw SymbolError(node->line, "Semantic Analysis: conflicting return type for function prototype '" + node->name + "'.");
        }
    }
}

void SemanticAnalysisVisitor::visit(ProcedurePrototypeNode* node) {
    // Convert parameter list to VarInfo (names may be empty for prototypes)
    std::vector<VarInfo> params;
    params.reserve(node->params.size());
    for (size_t i = 0; i < node->params.size(); ++i) {
        const auto &v = node->params[i];
        VarInfo param = v;
        param.type = resolveUnresolvedType(current_, param.type, node->line);
        params.push_back(param);
    }

    // Propagate resolved parameter types back to the AST node
    for (size_t i = 0; i < node->params.size() && i < params.size(); ++i) {
        node->params[i].type = params[i].type;
    }

    node->returnType = resolveUnresolvedType(current_, node->returnType, node->line);

    // Declare the procedure signature in the current (global) scope
    try {
        std::cerr << "declareProc: " << node->name << " line " << node->line << std::endl;
        current_->declareProc(node->name, params, node->returnType, node->line);
    } catch (...) {
        ProcInfo *existing = current_->resolveProc(node->name, params, node->line);
        if (existing->procReturn != node->returnType) {
            throw TypeError(
                node->line, "Semantic Analysis: conflicting return type for procedure prototype '" + node->name + "'.");
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
            throw SymbolError(node->line, "Semantic Analysis: duplicate parameter name '" + v.identifier + "' in function '" + node->name + "'.");
        }
        param.type = resolveUnresolvedType(current_, param.type, node->line);
        params.push_back(param);
    }

    // Propagate resolved parameter types back to the AST node
    for (size_t i = 0; i < node->parameters.size() && i < params.size(); ++i) {
        node->parameters[i].type = params[i].type;
    }

    node->returnType = resolveUnresolvedType(current_, node->returnType, node->line);

    // Declare or validate existing prototype declr
    try {
        current_->declareFunc(node->name, params, node->returnType, node->line);
    } catch (...) {
        FuncInfo* existing = current_->resolveFunc(node->name, params, node->line);
        if (existing->funcReturn != node->returnType) {
            throw std::runtime_error("Semantic Analysis: conflicting return type for function '" + node->name + "'.");
        }
    }

    // Enter function scope, bind parameters
    enterScopeFor(node, false, &node->returnType);
    current_->setInFunctionTrue(); 
    for (auto& v : params) {
        if (!v.isConst) { throw AssignError(node->line, "Non-const arg given as function parameter");}
        current_->declareVar(v.identifier, v.type, true, node->line);
    }

    // Analyze body
    if (!node->body) {
        throw std::runtime_error("Semantic Analysis: FATAL: function '" + node->name + "' missing body.");
    }
    node->body->accept(*this);

    // Ensure all paths return
    if (!guaranteesReturn(node->body.get())) {
        int errLine = computeMissingReturnLine(node->body.get());
        throw ReturnError(errLine, "Semantic Analysis: not all control paths return in function '" + node->name + "'.");
    }

    exitScope();
}

/* TODO add error line numbers */
void SemanticAnalysisVisitor::visit(ProcedureBlockNode* node) {
    // Resolve any alias-based return type first
    node->returnType = resolveUnresolvedType(current_, node->returnType, node->line);

    // Special case: main() constraints
    if (node->name == "main") {
        if (seenMain_) {
            throw SymbolError(node->line, "Semantic Analysis: Multiple definitions of procedure main().");
        }
        seenMain_ = true;
        if (!node->params.empty()) {
            throw MainError(node->line, "Semantic Analysis: procedure main() must not take parameters.");
        }
        if (node->returnType.baseType != BaseType::INTEGER) {
            throw MainError(node->line, "Incorrect return type for main procedure");
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
            throw SymbolError(node->line, std::string("Semantic Analysis: duplicate parameter name '") + v.identifier + "' in procedure '" + node->name + "'.");
        }
        param.type = resolveUnresolvedType(current_, param.type, node->line);
        params.push_back(param);
    }

    // Propagate resolved parameter types back to the AST node
    for (size_t i = 0; i < node->params.size() && i < params.size(); ++i) {
        node->params[i].type = params[i].type;
    }

    // Declare or validate existing declaration
    try {
        current_->declareProc(node->name, params, node->returnType, node->line);
    } catch (...) {
        ProcInfo* existing = current_->resolveProc(node->name, params, node->line);
        if (existing->procReturn != node->returnType) {
            throw TypeError(node->line, "Semantic Analysis: conflicting return type for procedure '" + node->name + "'.");
        }
    }

    // Enter procedure scope, bind params
    enterScopeFor(node, false, &node->returnType);
    for (auto& v : params) {
        current_->declareVar(v.identifier, v.type, v.isConst, node->line);
    }

    if (!node->body) {
        // should not happen
        throw std::runtime_error("Semantic Analysis: FATAL: procedure '" + node->name + "' missing body.");
    }
    node->body->accept(*this);

    // If non-void return expected, ensure all paths return
    if (node->returnType.baseType != BaseType::UNKNOWN) {
        if (!guaranteesReturn(node->body.get())) {
            int errLine = computeMissingReturnLine(node->body.get());
            throw ReturnError(errLine, "Semantic Analysis: not all control paths return in procedure '" + node->name + "'.");
        }
    }

    exitScope();
}

void SemanticAnalysisVisitor::visit(InferredDecNode* node) {
    if (!current_->isDeclarationAllowed()) {
        throw DefinitionError(node->line, "Semantic Analysis: Declarations must appear at the top of a block."); 
    }
    node->init->accept(*this);

    handleGlobalErrors(node);

    bool isConst = true;
    if (node->qualifier == "var") {
        isConst = false;
    } else if (node->qualifier == "const") {
    } else {
        throw std::runtime_error("Semantic Analysis: Invalid qualifier provided for type inference '" + node->qualifier + "'.");
    }

    CompleteType varType = resolveUnresolvedType(current_, node->init->type, node->line); // no need to check promotability


    // Ensure not already declared in scope
    current_->declareVar(node->name, varType, isConst, node->line);

    node->type = varType;
}

void SemanticAnalysisVisitor::visit(TupleTypedDecNode* node) {
    if (!current_->isDeclarationAllowed()) {
        throw DefinitionError(node->line, "Semantic Analysis: Declarations must appear at the top of a block."); 
    }
    if (node->init) {
        node->init->accept(*this);
    }
    // For tuple-typed declarations, the declared type is already present
    // on the declaration node as a CompleteType
    CompleteType varType = resolveUnresolvedType(current_, node->type, node->line);

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

    handleGlobalErrors(node);

    // Ensure not already declared in scope
    current_->declareVar(node->name, varType, isConst, node->line);

    // Ensure init expr type matches with var type (if provided)
    if (node->init != nullptr) {
        handleAssignError(node->name, varType, node->init->type, node->line);
    }

    node->type = varType;
}

void SemanticAnalysisVisitor::visit(StructTypedDecNode* node) {
    if (!current_->isDeclarationAllowed()) {
        throw DefinitionError(node->line, "Semantic Analysis: Declarations must appear at the top of a block."); 
    }

    // defer the handling of init because we must first resolve the struct type name.

    // Resolve underlying struct member types (may include aliases).
    CompleteType structType = resolveUnresolvedType(current_, node->type, node->line);
    if (structType.baseType != BaseType::STRUCT) {
        throw std::runtime_error("Semantic Analysis: FATAL: StructTypedDecNode with non-struct type.");
    }

    bool isConst = true;
    if (node->qualifier == "var") {
        isConst = false;
    } else if (node->qualifier == "const" || node->qualifier.empty()) {
        // do nothing
    } else {
        throw std::runtime_error(
            "Semantic Analysis: Invalid qualifier provided for struct declaration '" +
            node->qualifier + "'.");
    }

    // Declare struct name and save as alias to be used later
    current_->declareStructType(structType.aliasName, structType, node->line);


    // visit initializer once struct type declared
    if (node->init) {
        node->init->accept(*this);
    }


    // Optional: declare a variable of this struct type if a name was provided
    if (!node->name.empty()) {

        // Allow global struct TYPE declrs, but disallow global struct vars
        handleGlobalErrors(node);
        // Ensure not already declared in this scope
        current_->declareVar(node->name, structType, isConst, node->line);

        if (node->init != nullptr) {
            CompleteType initType =
                resolveUnresolvedType(current_, node->init->type, node->line);
            handleAssignError(node->name, structType, initType, node->line);
        }
    }

    node->type = structType;
}

void SemanticAnalysisVisitor::visit(TypeAliasDecNode* node) {
    if (!current_->isInGlobal()) {
        throw StatementError(node->line, "Alias declaration in non-global scope '" + node->alias + "'.");
    }
    // Resolve aliased type. Support aliasing built-ins or another alias.
    CompleteType aliased = node->type;
    if (aliased.baseType == BaseType::UNKNOWN && !node->declTypeName.empty()) {
        // Try resolving as another alias name
        try {
            aliased = *current_->resolveAlias(node->declTypeName, node->line);
        } catch (...) {
            // If not an alias, leave as UNKNOWN; builder should have set to built-in type
        }
    }
    current_->declareAlias(node->alias, aliased, node->line);

    // assume node has been initialized with correct type if not UNKNOWN
}

void SemanticAnalysisVisitor::visit(TypeAliasNode *node) {
    if (!current_) {
        throw std::runtime_error("Semantic Analysis: TypeAliasNode visited with no current scope.");
    }

    if (!node->aliasName.empty()) {
        // Treat aliasName as an unresolved type name that may refer to either
        // a basic type-alias or a struct type in the current scope chain.
        CompleteType unresolved(node->aliasName); // BaseType::UNRESOLVED
        node->type = resolveUnresolvedType(current_, unresolved, node->line);
        return;
    }

    // If no alias name, assume node->type already holds a concrete type.
}

void SemanticAnalysisVisitor::visit(TupleTypeAliasNode *node) {
    // This node represents a declaration: TYPEALIAS tuple_dec ID
    if (!current_->isInGlobal()) {
        throw StatementError(node->line, "Alias declaration in non-global scope '" + node->aliasName + "'.");
    }
    // Resolve any unresolved subtypes in the tuple type before storing the alias
    CompleteType resolvedType = resolveUnresolvedType(current_, node->type, node->line);
    current_->declareAlias(node->aliasName, resolvedType, node->line);
}

void SemanticAnalysisVisitor::visit(AssignStatNode* node) {
    node->expr->accept(*this);
    // std::cerr << "Visiting AssignNode";
    // handles if undeclared
    VarInfo* varInfo = current_->resolveVar(node->name, node->line);
    
    if (varInfo->isConst) {
        throw AssignError(node->line, "Semantic Analysis: cannot assign to const variable '" + node->name + "'."); // TODO add line num
    }

    CompleteType varType = resolveUnresolvedType(current_, varInfo->type, node->line);
    CompleteType exprType = resolveUnresolvedType(current_, node->expr->type, node->line);

    handleAssignError(node->name, varType, exprType, node->line);

    node->type = varType;
}

void SemanticAnalysisVisitor::visit(DestructAssignStatNode* node) {
    if (!node->expr) {
        throw AssignError(node->line, "Semantic Analysis: missing expression in destructuring assignment.");
    }

    // Analyse RHS first to know its tuple shape
    node->expr->accept(*this);
    CompleteType rhsType = resolveUnresolvedType(current_, node->expr->type, node->line);

    if (rhsType.baseType != BaseType::TUPLE) {
        throw AssignError(node->line, "Semantic Analysis: destructuring assignment requires a tuple expression on the right-hand side.");
    }

    if (rhsType.subTypes.size() != node->names.size()) {
        throw AssignError(node->line, "Semantic Analysis: tuple arity mismatch in destructuring assignment.");
    }

    // Check each target variable
    for (size_t i = 0; i < node->names.size(); ++i) {
        const std::string &name = node->names[i];
        VarInfo* varInfo = current_->resolveVar(name, node->line);
        if (!varInfo) {
            throw SymbolError(node->line, "Semantic Analysis: variable '" + name + "' not defined in destructuring assignment.");
        }
        if (varInfo->isConst) {
            throw AssignError(node->line, "Semantic Analysis: cannot assign to const variable '" + name + "' in destructuring assignment.");
        }

        CompleteType varType = resolveUnresolvedType(current_, varInfo->type, node->line);
        CompleteType elemType = resolveUnresolvedType(current_, rhsType.subTypes[i], node->line);
        handleAssignError(name, varType, elemType, node->line);
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
        throw StatementError(node->line, "Cannot use 'break' outside of loop."); 
    }
}

void SemanticAnalysisVisitor::visit(ContinueStatNode* node) {
    if (!current_->isInLoop()) {
        throw StatementError(node->line, "Cannot use 'continue' outside of loop.");
    }
}

void SemanticAnalysisVisitor::visit(ReturnStatNode* node) {
    // Allow return inside func/proc, determined by non-null expected return type on the scope
    if (current_->getReturnType() == nullptr) {
        throw StatementError(node->line, "Cannot use 'return' outside of function."); 
    }

    // If expression provided, type-check against expected return type
    if (node->expr) {
        node->expr->accept(*this);
        handleAssignError("", *current_->getReturnType(), node->expr->type, node->line);
    } else {
        // No value returned: only legal if the declared return type is 'void' equivalent
        if (current_->getReturnType()->baseType != BaseType::UNKNOWN) {
            throw TypeError(node->line, "Semantic Analysis: Non-void return required by declaration.");
        }
    }
}

void SemanticAnalysisVisitor::visit(CallStatNode* node) {
    // The CallStatNode wraps an expression-style FuncCallExprOrStructLiteral in `call`.
    if (!node->call) {
        throw std::runtime_error("Semantic Analysis: FATAL: empty call statement");
    }
    if (current_->isInFunction()) {
        throw CallError(node->line, "Cannot call procedure inside a function.");
    }

    std::vector<VarInfo> args;
    args.reserve(node->call->args.size());
    for (const auto& e : node->call->args) {
        if (e) e->accept(*this);
        CompleteType argType = e ? resolveUnresolvedType(current_, e->type, node->line)
                                 : CompleteType(BaseType::UNKNOWN);
        args.push_back(VarInfo{"", argType, true});
    }

    // Resolve as procedure only. prevent calling a function via 'call'
    try {
        (void) current_->resolveProc(node->call->funcName, args, node->line);
    } catch (...) {
        throw SymbolError(node->line, "Semantic Analysis: Unknown procedure '" + node->call->funcName + "' in call statement.");
    }

    // Statements have no resultant type
    node->type = CompleteType(BaseType::UNKNOWN);
}

void SemanticAnalysisVisitor::visit(TupleAccessAssignStatNode* node) {
    if (!node->target || !node->expr) {
        throw AssignError(node->line, "Semantic Analysis: malformed tuple access assignment.");
    }

    // Analyse LHS tuple access first (binds tuple variable + element type)
    node->target->accept(*this);

    if (!node->target->binding) {
        throw std::runtime_error("Semantic Analysis: FATAL: TupleAccessNode missing binding in assignment.");
    }

    VarInfo* tupleVar = node->target->binding;
    if (tupleVar->isConst) {
        throw AssignError(node->line, "Semantic Analysis: cannot assign to element of const tuple '" +
                                 node->target->tupleName + "'.");
    }

    // visit rhs
    node->expr->accept(*this);

    CompleteType elemType = resolveUnresolvedType(current_, node->target->type, node->line);
    CompleteType exprType = resolveUnresolvedType(current_, node->expr->type, node->line);

    handleAssignError(node->target->tupleName, elemType, exprType, node->line);

    node->type = CompleteType(BaseType::UNKNOWN);
}

void SemanticAnalysisVisitor::visit(StructAccessAssignStatNode *node) {
    if (!node->target || !node->expr) {
        throw AssignError(node->line, "Semantic Analysis: malformed struct access assignment.");
    }


    // Visit lhs and get type
    node->target->accept(*this);
    if (!node->target->binding) {
        throw std::runtime_error("Semantic Analysis: FATAL: StructAccessNode missing binding in assignment.");
    }
    VarInfo *structVar = node->target->binding;
    if (structVar->isConst) throw AssignError(node->line, "Semantic Analysis: cannot assign to field of const struct '" + node->target->structName + "." + node->target->fieldName + "'");

    // Visit rhs
    node->expr->accept(*this);

    CompleteType elemType = resolveUnresolvedType(current_, node->target->type, node->line);
    CompleteType exprType = resolveUnresolvedType(current_, node->expr->type, node->line);

    handleAssignError(node->target->structName, elemType, exprType, node->line);

    node->type = CompleteType(BaseType::UNKNOWN);
}

void SemanticAnalysisVisitor::visit(ArrayAccessAssignStatNode* node) {
    if (!node->target || !node->expr) {
        throw std::runtime_error("ArrayAccessAssignStat node: missing target or expression");
    }

    // 1. Visit the target to set its binding and determine its type
    node->target->accept(*this);

    if (!node->target->binding) {
        // This check is good practice, though accept() should have set it or thrown.
        throw std::runtime_error("Semantic Analysis: FATAL: ArrayAccessNode missing binding in assignment.");
    }
    
    // 2. Check for assignment to const array
    VarInfo* arrayVar = node->target->binding;
    if (arrayVar->isConst) {
        throw AssignError(node->line, "Semantic Analysis: cannot assign to element of const array '" +
                                 node->target->id + "'.");
    }

    // 3. Visit the expression on the RHS
    node->expr->accept(*this);

    // 4. Type check the assignment
    CompleteType elemType = resolveUnresolvedType(current_, node->target->type, node->line);
    CompleteType exprType = resolveUnresolvedType(current_, node->expr->type, node->line);
    
    handleAssignError(node->target->id, elemType, exprType, node->line);

    node->type = CompleteType(BaseType::UNKNOWN);
}

void SemanticAnalysisVisitor::visit(FuncCallExprOrStructLiteral* node) {
    // Evaluate argument expressions and build a signature to resolve the callee
    std::vector<VarInfo> args;
    args.reserve(node->args.size());
    for (const auto& e : node->args) {
        if (e) e->accept(*this);
        CompleteType argType = e ? resolveUnresolvedType(current_, e->type, node->line)
                                 : CompleteType(BaseType::UNKNOWN);
        args.push_back(VarInfo{"", argType, true});
        
    }

    // Try resolving as function
    // -----------------------
    FuncInfo* finfo = nullptr;
    try {
        finfo = current_->resolveFunc(node->funcName, args, node->line);
    } catch (...) {
        finfo = nullptr;
    }
    if (finfo) {
        std::cerr << "Function resolved";
        // Function call in expression
        node->type = finfo->funcReturn;
        node->resolvedFunc = *finfo; // cache resolved info for later passes
        node->callType = CallType::FUNCTION;
        return;
    }

    // Then try resolving as procedure
    // -----------------------
    ProcInfo* pinfo = nullptr;
    try {
        pinfo = current_->resolveProc(node->funcName, args, node->line);
    } catch (...) {
        pinfo = nullptr;
    }

    if (pinfo) {
        // Procedures may have a return type; only those may appear in expressions.
        if (pinfo->procReturn.baseType == BaseType::UNKNOWN) {
            throw TypeError(node->line, "Semantic Analysis: procedure '" + node->funcName +
                                  "' used as expression but has no return type.");
        }
        node->type = pinfo->procReturn;
        node->callType = CallType::PROCEDURE;
        return;
    }

    // Try resolving as struct literal (struct constructor call)
    // -----------------------
    CompleteType* structType = nullptr;
    try {
        structType = current_->resolveStructType(node->funcName, node->line);
    } catch (...) {
        structType = nullptr;
    }
    if (structType) {
        node->type = *structType;

        // If the struct type tracks element subtypes, validate arity and types
        if (structType->subTypes.empty()) throw std::runtime_error("SemanticAnalysis::FuncCallExprOrStructLiteral: Empty subtpyes for struct literal.");
        if (node->args.size() != structType->subTypes.size()) {
            throw TypeError(
                node->line,
                "Semantic Analysis: Struct literal for '" + node->funcName +
                    "' has argument count mismatch. Expected " +
                    std::to_string(structType->subTypes.size()) + ", got " +
                    std::to_string(node->args.size()) + ".");
        }
        for (size_t i = 0; i < node->args.size(); ++i) {
            handleAssignError(
                node->funcName + "." + std::to_string(i),
                structType->subTypes[i],
                node->args[i]->type,
                node->line);
        }
        node->callType = CallType::STRUCT_LITERAL;
        return;
    }


    throw SymbolError(node->line, "Semantic Analysis: Unknown function/procedure/struct '" +
                            node->funcName + "' in expression.");
}

void SemanticAnalysisVisitor::visit(IfNode* node) {
    // Evaluate and type-check condition first
    node->cond->accept(*this);
    if (node->cond->type.baseType != BaseType::BOOL) {
        throw TypeError(node->line, "Semantic Analysis: if condition must be boolean; got '" + toString(node->cond->type) + "'.");
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

    // Visit declarations, then statements
    for (const auto& d : node->decs) {
        if (d) d->accept(*this);
    }
    current_->disableDeclarations();
    for (const auto& s : node->stats) {
        if (s) s->accept(*this);
    }

    exitScope();
}

void SemanticAnalysisVisitor::visit(LoopNode* node) {
    // Optional condition must be boolean
    if (node->cond) {
        node->cond->accept(*this);
        if (node->cond->type.baseType != BaseType::BOOL) {
            throw TypeError(node->line, "Semantic Analysis: loop condition must be boolean; got '" + toString(node->cond->type) + "'.");
        }
    }
    // Enter loop scope so 'break'/'continue' are legal
    enterScopeFor(node, true, current_->getReturnType());
    if (node->body) {
        node->body->accept(*this);
    }
    exitScope();
}

void SemanticAnalysisVisitor::visit(IteratorLoopNode* node) {
    if (!node->domainExpr) {
        throw std::runtime_error("IteratorLoop: missing domain expression");
    }
    if (!node->body) {
        throw std::runtime_error("IteratorLoop: missing loop body");
    }

    auto range = std::dynamic_pointer_cast<RangeExprNode>(node->domainExpr);
    if (!range) {
        throw TypeError(node->line, "Iterator loop currently supports only range domains.");
    }

    // Prepare start/end/step expressions (default start to 1 if omitted, step to 1)
    std::shared_ptr<ExprNode> startExpr = range->start;
    std::shared_ptr<ExprNode> endExpr = range->end;
    std::shared_ptr<ExprNode> stepExpr = range->step;
    if (!startExpr) {
        startExpr = std::make_shared<IntNode>(1);
        startExpr->line = node->line;
    }
    if (!endExpr) {
        throw TypeError(node->line, "Iterator loop range requires an end expression.");
    }
    if (!stepExpr) {
        stepExpr = std::make_shared<IntNode>(1);
        stepExpr->line = node->line;
    }

    // Type-check bounds and stride
    startExpr->accept(*this);
    endExpr->accept(*this);
    stepExpr->accept(*this);
    if (startExpr->type.baseType != BaseType::INTEGER) {
        throw TypeError(node->line, "Iterator loop range start must be integer.");
    }
    if (endExpr->type.baseType != BaseType::INTEGER) {
        throw TypeError(node->line, "Iterator loop range end must be integer.");
    }
    if (stepExpr->type.baseType != BaseType::INTEGER) {
        throw TypeError(node->line, "Iterator loop range stride must be integer.");
    }

    // Reject non-positive stride
    if (auto stepInt = std::dynamic_pointer_cast<IntNode>(stepExpr)) {
        if (stepInt->value <= 0) {
            throw StrideError(node->line, "Iterator loop range stride must be positive.");
        }
    }

    // Hidden temp names
    static int loopTempCounter = 0;
    const std::string startName = "__loop_start_" + std::to_string(loopTempCounter);
    const std::string endName   = "__loop_end_" + std::to_string(loopTempCounter);
    const std::string idxName   = "__loop_idx_" + std::to_string(loopTempCounter);
    const std::string stepName  = "__loop_step_" + std::to_string(loopTempCounter);
    loopTempCounter++;

    // Declarations for start/end/step/idx
    auto startDec = std::make_shared<InferredDecNode>(startName, "const", startExpr);
    startDec->line = node->line;
    auto endDec = std::make_shared<InferredDecNode>(endName, "const", endExpr);
    endDec->line = node->line;
    auto stepDec = std::make_shared<InferredDecNode>(stepName, "const", stepExpr);
    stepDec->line = node->line;

    auto idxInit = std::make_shared<IdNode>(startName);
    idxInit->line = node->line;
    auto idxDec = std::make_shared<InferredDecNode>(idxName, "var", idxInit);
    idxDec->line = node->line;

    // While condition: idx <= end
    auto idxIdForCond = std::make_shared<IdNode>(idxName);
    idxIdForCond->line = node->line;
    auto endIdForCond = std::make_shared<IdNode>(endName);
    endIdForCond->line = node->line;
    auto condExpr = std::make_shared<CompExpr>("<=", idxIdForCond, endIdForCond);
    condExpr->line = node->line;

    // Iterator binding: const iter = idx
    auto idxIdForIter = std::make_shared<IdNode>(idxName);
    idxIdForIter->line = node->line;
    auto iterDec = std::make_shared<InferredDecNode>(node->iterName, "const", idxIdForIter);
    iterDec->line = node->line;

    // idx = idx + step
    auto idxIdForInc = std::make_shared<IdNode>(idxName);
    idxIdForInc->line = node->line;
    auto stepIdForInc = std::make_shared<IdNode>(stepName);
    stepIdForInc->line = node->line;
    auto incExpr = std::make_shared<AddExpr>("+", idxIdForInc, stepIdForInc);
    incExpr->line = node->line;
    auto incStat = std::make_shared<AssignStatNode>(idxName, incExpr);
    incStat->line = node->line;

    // While body: iterator binding + original body
    std::vector<std::shared_ptr<DecNode>> bodyDecs;
    std::vector<std::shared_ptr<StatNode>> bodyStats;
    bodyDecs.push_back(iterDec);
    if (node->body) {
        // Preserve existing declarations/statements
        bodyDecs.insert(bodyDecs.end(), node->body->decs.begin(), node->body->decs.end());
        bodyStats.insert(bodyStats.end(), node->body->stats.begin(), node->body->stats.end());
    }
    bodyStats.push_back(incStat);
    auto whileBody = std::make_shared<BlockNode>(std::move(bodyDecs), std::move(bodyStats));
    whileBody->line = node->line;

    auto whileNode = std::make_shared<LoopNode>(whileBody, condExpr);
    whileNode->kind = LoopKind::While;
    whileNode->line = node->line;

    // Outer block: declare temps then execute while
    std::vector<std::shared_ptr<DecNode>> outerDecs;
    outerDecs.push_back(startDec);
    outerDecs.push_back(endDec);
    outerDecs.push_back(stepDec);
    outerDecs.push_back(idxDec);
    std::vector<std::shared_ptr<StatNode>> outerStats;
    outerStats.push_back(whileNode);
    auto lowered = std::make_shared<BlockNode>(std::move(outerDecs), std::move(outerStats));
    lowered->line = node->line;

    // Store lowered form for downstream passes
    node->lowered = lowered;

    // Visit lowered form
    lowered->accept(*this);
    node->type = CompleteType(BaseType::UNKNOWN);
}

void SemanticAnalysisVisitor::visit(GeneratorExprNode* node) {
    // Arity check: only 1D for now
    if (node->domains.empty() || node->domains.size() > 2) {
        throw TypeError(node->line, "Generator must have 1 or 2 domain variables.");
    }

    // For 1D only in this milestone
    size_t arity = node->domains.size();
    if (arity != 1) {
        throw TypeError(node->line, "2D generators not yet implemented.");
    }

    // Domain normalization: visit domain expr (allows range, array, nested generator)
    auto &domPair = node->domains[0];
    if (!domPair.second) {
        throw TypeError(node->line, "Generator domain is null.");
    }
    domPair.second->accept(*this);
    CompleteType domType = resolveUnresolvedType(current_, domPair.second->type, node->line);
    if (domType.baseType != BaseType::ARRAY && domType.baseType != BaseType::VECTOR && domType.baseType != BaseType::MATRIX && domType.baseType != BaseType::UNKNOWN) {
        throw TypeError(node->line, "Generator domain must be array/vector/matrix or range.");
    }

    // Determine length for 1D
    int len = -1; // -1 dynamic
    if (!domType.dims.empty() && domType.dims[0] >= 0) {
        len = domType.dims[0];
    }
    // If domain is RangeExprNode, try to compute static len; else keep dynamic
    if (auto rangeDom = std::dynamic_pointer_cast<RangeExprNode>(domPair.second)) {
        // Best-effort static length if start/end/step are int literals
        int startLit = 1;
        int endLit = -1;
        int stepLit = 1;
        bool startIsLit = false, endIsLit = false, stepIsLit = false;
        if (rangeDom->start) {
            if (auto s = std::dynamic_pointer_cast<IntNode>(rangeDom->start)) { startLit = s->value; startIsLit = true; }
        } else { startIsLit = true; startLit = 1; }
        if (rangeDom->end) {
            if (auto e = std::dynamic_pointer_cast<IntNode>(rangeDom->end)) { endLit = e->value; endIsLit = true; }
        }
        if (rangeDom->step) {
            if (auto st = std::dynamic_pointer_cast<IntNode>(rangeDom->step)) { stepLit = st->value; stepIsLit = true; }
        } else { stepIsLit = true; stepLit = 1; }
        if (stepLit <= 0 && stepIsLit) throw StrideError(node->line, "Iterator loop range stride must be positive.");
        if (startIsLit && endIsLit && stepIsLit) {
            if (endLit < startLit) len = 0;
            else len = static_cast<int>((static_cast<int64_t>(endLit - startLit) / stepLit) + 1);
        }
    }

    // Determine element type: use domain element type where possible
    CompleteType iterType(BaseType::UNKNOWN);
    if (domType.baseType == BaseType::ARRAY || domType.baseType == BaseType::VECTOR || domType.baseType == BaseType::MATRIX) {
        if (!domType.subTypes.empty()) iterType = domType.subTypes[0];
    } else {
        // Range domain -> integer
        iterType = CompleteType(BaseType::INTEGER);
    }

    // Bind iterator with inferred element type in a child scope, then visit RHS to refine
    enterScopeFor(node, current_->isInLoop(), current_->getReturnType());
    current_->declareVar(domPair.first, iterType, true, node->line);
    if (node->rhs) node->rhs->accept(*this);
    CompleteType elemType = resolveUnresolvedType(current_, node->rhs ? node->rhs->type : CompleteType(BaseType::UNKNOWN), node->line);
    exitScope();

    // Build result type
    CompleteType resultType(BaseType::ARRAY);
    resultType.subTypes.push_back(elemType);
    resultType.dims = {len};
    node->type = resultType;

    // Allocate result var (in lowered block) with computed len (static or dynamic)
    static int genTempCounter = 0;
    std::string resName = "__gen_tmp_" + std::to_string(genTempCounter++);
    auto resDec = std::make_shared<ArrayTypedDecNode>("var", resName, resultType);
    resDec->line = node->line;
    // Declare in current scope so downstream passes can resolve it
    current_->declareVar(resName, resultType, false, node->line);

    // Build iterator loop to fill result: loop idx in domain { res[w]=rhs; w++ }
    // Write index declaration (var w = 1)
    auto one = std::make_shared<IntNode>(1); one->line = node->line;
    auto wName = "__gen_w_" + std::to_string(genTempCounter);
    auto wDec = std::make_shared<InferredDecNode>(wName, "var", one);
    wDec->line = node->line;
    current_->declareVar(wName, CompleteType(BaseType::INTEGER), false, node->line);

    // Iterator binding inside loop: const iter = domain element (handled by IteratorLoopNode)
    // Body: res[w] = rhs; w = w + 1;
    // Build assign res[w] = rhs
    auto resId = std::make_shared<IdNode>(resName); resId->line = node->line;
    auto wIdForStore = std::make_shared<IdNode>(wDec->name); wIdForStore->line = node->line;
    auto arrAccess = std::make_shared<ArrayAccessNode>(resId->id, wIdForStore);
    arrAccess->line = node->line;
    // RHS: use original rhs with iterator binding resolved in loop scope; just reuse node->rhs
    auto storeStat = std::make_shared<ArrayAccessAssignStatNode>(arrAccess, node->rhs);
    storeStat->line = node->line;

    // Increment w
    auto wIdForInc = std::make_shared<IdNode>(wDec->name); wIdForInc->line = node->line;
    auto incExpr = std::make_shared<AddExpr>("+", wIdForInc, one);
    incExpr->line = node->line;
    auto incStat = std::make_shared<AssignStatNode>(wDec->name, incExpr);
    incStat->line = node->line;

    // Loop body block: store RHS into result[w], then increment w
    std::vector<std::shared_ptr<DecNode>> loopDecs;
    std::vector<std::shared_ptr<StatNode>> loopStats;
    loopStats.push_back(storeStat);
    loopStats.push_back(incStat);
    auto loopBody = std::make_shared<BlockNode>(std::move(loopDecs), std::move(loopStats));
    loopBody->line = node->line;

    // Iterator loop over domain
    auto iterLoop = std::make_shared<IteratorLoopNode>(domPair.first, domPair.second, loopBody);
    iterLoop->line = node->line;

    // Lowered block: [resDec, wDec] then iter loop
    std::vector<std::shared_ptr<DecNode>> loweredDecs;
    loweredDecs.push_back(resDec);
    loweredDecs.push_back(wDec);
    std::vector<std::shared_ptr<StatNode>> loweredStats;
    loweredStats.push_back(iterLoop);
    auto lowered = std::make_shared<BlockNode>(std::move(loweredDecs), std::move(loweredStats));
    lowered->line = node->line;

    node->lowered = lowered;
    node->loweredResultName = resName;

    // Visit lowered STATS ONLY to resolve references without creating new scope/shadowing vars
    // We skip visiting decs in SA because we manually declared them in the current scope above.
    // MLIRGen will still visit the declarations to handle allocation/initialization.
    for (const auto& s : lowered->stats) {
        if (s) s->accept(*this);
    }
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
        throw TypeError(node->line,"Semantic Analysis error: Applying operator '" + node->op + "' to operand with type UNKNOWN.");
    }

    std::string op = node->op;
    if (op == "-" || op == "+") {
        // Not permitted: bool, character, tuple, struct, string
        // permitted: int, real, arrays(int, real), vector(int, real), matrix(int, real)
        const BaseType illegalTypes[] = {BaseType::BOOL, BaseType::CHARACTER, BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING};

        if (std::find(std::begin(illegalTypes), std::end(illegalTypes), node->operand->type.baseType) != std::end(illegalTypes)) {
            //throw TypeError(node->line, "Operand Error");
            throwOperandError(op, {node->operand->type}, "", node->line);
        }

    } else if (op == "not") {
        // Not permitted: character, int, real, tuple, struct, string
        // permitted: bool, arrays(bool), vector(bool), matrices(bool)
        const BaseType illegalTypes[] = {BaseType::CHARACTER, BaseType::INTEGER, BaseType::REAL, BaseType::TUPLE, BaseType::STRUCT, BaseType::STRING};

        if (std::find(std::begin(illegalTypes), std::end(illegalTypes), node->operand->type.baseType) != std::end(illegalTypes)) {
            //throw TypeError(node->line, "Operand Error");
            throwOperandError(op, {node->operand->type}, "", node->line);
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
        throwOperandError("^", {leftOperandType}, "Illegal left operand", node->line);
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError("^", {rightOperandType}, "Illegal right operand", node->line);
    }

    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        throw TypeError(node->line, "No promotion possible between operands");
        //throwOperandError("^", {leftOperandType, rightOperandType}, "No promotion possible between operands");
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
        throw TypeError(node->line, "Illegal left operand");
        // throwOperandError(node->op, {leftOperandType}, "Illegal left operand");
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throw TypeError(node->line, "Illegal right operand");
        // throwOperandError(node->op, {rightOperandType}, "Illegal right operand");
    }

    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        throw TypeError(node->line, "No promotion possible between operands");
        // throwOperandError(node->op, {leftOperandType, rightOperandType}, "No promotion possible between operands");
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
        throwOperandError(node->op, {leftOperandType}, "Illegal left operand", node->line);
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {rightOperandType}, "Illegal right operand", node->line);
    }
    // ! THIS IS A STUB METHOD, COMPILE TIME SIZE SHOULD BE EVALUATED IN CONSTANT FOLDING
    // If both are arrays, attempt to compare compile-time sizes.
    // Note: CompleteType carries dimension sizes in dims, so inspect the
    // expression nodes for compile-time size info: IdNode -> type.dims[0],
    // ArrayLiteralNode -> literal element count. If both sizes are known and
    // differ, throw a SizeError.
    if (leftOperandType.baseType == BaseType::ARRAY && rightOperandType.baseType == BaseType::ARRAY) {
        auto getCompileTimeSize = [&](std::shared_ptr<ExprNode> expr) -> std::optional<int64_t> {
            if (!expr) return std::nullopt;
            // IdNode: check binding's VarInfo type dims
            if (auto idn = std::dynamic_pointer_cast<IdNode>(expr)) {
                if (idn->binding && !idn->binding->type.dims.empty() &&
                    idn->binding->type.dims[0] >= 0) {
                    return static_cast<int64_t>(idn->binding->type.dims[0]);
                }
                return std::nullopt;
            }
            // Array literal: size is the number of elements (if list present)
            if (auto lit = std::dynamic_pointer_cast<ArrayLiteralNode>(expr)) {
                if (lit->list) return static_cast<int64_t>(lit->list->list.size());
                return 0;
            }
            // Other expressions: cannot determine compile-time size
            return std::nullopt;
        };

        auto lsz = getCompileTimeSize(node->left);
        auto rsz = getCompileTimeSize(node->right);
        if (lsz.has_value() && rsz.has_value() && lsz.value() != rsz.value()) {
            throw SizeError(node->line, "Semantic Analysis: Arrays must have same size");
        }
    }
    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        throw TypeError(node->line, "No promotion possible between operands");
        //throwOperandError(node->op, {leftOperandType, rightOperandType}, "No promotion possible between operands");
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
        throwOperandError(node->op, {leftOperandType}, "Illegal left operand", node->line);
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {rightOperandType}, "Illegal right operand", node->line);
    }

    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        
       // throwOperandError(node->op, {leftOperandType, rightOperandType}, "No promotion possible between operands");
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
        throwOperandError("not", {node->operand->type}, "", node->line);
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
    VarInfo* varInfo = current_->resolveVar(node->id, node->line); // handles no-declr
    node->type = varInfo->type;
    node->binding = varInfo;
}

void SemanticAnalysisVisitor::visit(TupleLiteralNode* node) {
    CompleteType literalType = CompleteType(BaseType::TUPLE);
    literalType.subTypes.reserve(node->elements.size());

    if (node->elements.size() < 2) {
        throw LiteralError(node->line, "All tuples must have at least 2 elements, not " + std::to_string(node->elements.size()) + ".");
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
    VarInfo* varInfo = current_->resolveVar(node->tupleName, node->line);
    
    if (!varInfo) {
        throw std::runtime_error("Semantic Analysis: FATAL: Variable '" + node->tupleName + "' not found in TupleAccessNode");
    }

    if (varInfo->type.baseType != BaseType::TUPLE) {
        throw std::runtime_error("Semantic Analysis: FATAL: Non-tuple type '" + toString(varInfo->type) + "' in TupleAccessNode");
    }

    if (node->index > varInfo->type.subTypes.size() || node->index == 0) {
        throw SizeError(node->line, "Index " + std::to_string(node->index) + " out of range for tuple of len " + std::to_string(varInfo->type.subTypes.size()));
        return; 
    }

    // Bind this access node to the underlying tuple variable so codegen
    // doesn't need to re-resolve names and can honour declaration order.
    node->binding = varInfo;
    node->type = varInfo->type.subTypes[node->index - 1];
}

void SemanticAnalysisVisitor::visit(StructAccessNode* node) {
    VarInfo* varInfo = current_->resolveVar(node->structName, node->line);
    if (!varInfo) {
        throw std::runtime_error("Semantic Analysis: FATAL: Variable '" + node->structName + "' not found in StructAccessNode");
    }

    if (varInfo->type.baseType != BaseType::STRUCT) {
        throw std::runtime_error("Semantic Analysis: FATAL: Non-struct type '" + toString(varInfo->type) + "' in StructAccessNode");
    }

    const auto fieldNames = varInfo->type.fieldNames;
    if (std::find(fieldNames.begin(), fieldNames.end(), node->fieldName) == fieldNames.end()) 
    throw SymbolError(node->line, "Field '" + node->fieldName + "' not found in struct" + node->structName + "."); 

    // Get field index as integer
    const size_t fieldIndex = std::distance(fieldNames.begin(), std::find(fieldNames.begin(), fieldNames.end(), node->fieldName));
    node->fieldIndex = fieldIndex;
    node->type = varInfo->type.subTypes[fieldIndex];
    node->binding = varInfo;
}

void SemanticAnalysisVisitor::visit(TypeCastNode* node) {
    // Evaluate operand first
    node->expr->accept(*this);
    // Resolve target type: prefer explicit alias name when present; otherwise
    // use the concrete base type provided by the AST.
    CompleteType target(BaseType::UNKNOWN);
    if (!node->targetAliasName.empty()) {
        target = *current_->resolveAlias(node->targetAliasName, node->line);
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
        throw TypeError(node->line, std::string("Semantic Analysis: cannot cast from '") + toString(node->expr->type) + "' to '" + toString(target) + "'.");
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
        throw TypeError(node->line, std::string("Semantic Analysis: cannot cast from '") + toString(node->expr->type) + "' to '" + toString(node->targetTupleType) + "'.");
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
        throwOperandError(node->op, {leftOperandType}, "Illegal left operand", node->line);
    }
    if (std::find(std::begin(illegalTypes), std::end(illegalTypes), rightOperandType.baseType) != std::end(illegalTypes)) {
        throwOperandError(node->op, {rightOperandType}, "Illegal right operand", node->line);
    }

    // Struct comparisons
    if (leftOperandType.baseType == BaseType::STRUCT ||
        rightOperandType.baseType == BaseType::STRUCT) {

        // cannot compare struct with non-struct.
        if (leftOperandType.baseType != BaseType::STRUCT || rightOperandType.baseType != BaseType::STRUCT) {
            throwOperandError(node->op, {leftOperandType, rightOperandType},"Cannot compare struct with non-struct", node->line);
        }

        // structs of different subtypes cannot be compared
        if (promote(leftOperandType, rightOperandType) != rightOperandType) {
            throwOperandError(node->op, {leftOperandType, rightOperandType},"Cannot compare struct of different subtypes", node->line);
        }

        node->type = CompleteType(BaseType::BOOL);
        return;
    }

    CompleteType finalType = promote(leftOperandType, rightOperandType);
    if (finalType.baseType == BaseType::UNKNOWN) {
        finalType = promote(rightOperandType, leftOperandType);
    }

    if (finalType.baseType == BaseType::UNKNOWN) {
        throwOperandError(node->op, {leftOperandType, rightOperandType}, "No promotion possible between operands", node->line);
    }

    node->type = CompleteType(BaseType::BOOL);
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
        throwOperandError(node->op, {leftOperandType, rightOperandType}, "Illegal operand(s) in 'and' expr.", node->line);
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
        throwOperandError(node->op, {leftOperandType, rightOperandType}, "Illegal operand(s) in or/xor expr.", node->line);
    }

    node->type = BaseType::BOOL;
}

void SemanticAnalysisVisitor::throwOperandError(const std::string op, const std::vector<CompleteType>& operands, std::string additionalInfo, int line) {
    std::stringstream ss;
    ss << "Semantic Analysis: Applying operator '" << op << "' to operand";

    if (operands.size() > 1) ss << "s";
    ss << ": ";

    for (size_t i = 0; i < operands.size(); ++i) {
        ss << "'" << toString(operands[i]) << "'";
        if (i < operands.size() - 1) ss << ", ";
    }

    if (!additionalInfo.empty()) {
        ss << "\n" + additionalInfo;
    }

    throw TypeError(line, ss.str());
}


/*
If empty string provided, prints non-variable promotion error msg
*/
void SemanticAnalysisVisitor::handleAssignError(const std::string varName, const CompleteType &varType, const CompleteType &exprType, int line) {
    // Normalise any alias-based types before checking compatibility.
    CompleteType resolvedVarType = resolveUnresolvedType(current_, varType, line);
    CompleteType resolvedExprType = resolveUnresolvedType(current_, exprType, line);

    // Encapsulate the type compatibility check here
    if (promote(resolvedExprType, resolvedVarType) != resolvedVarType) {
        if (varName != "") {
            TypeError err(
                line,
                std::string("Semantic Analysis: Cannot assign type '") + toString(resolvedExprType) +
                "' to variable '" + varName + "' of type '" + toString(resolvedVarType) + "'."
            );
            throw err;
        } else {
            TypeError err(
                line,
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

void SemanticAnalysisVisitor::handleGlobalErrors(DecNode *node) {
    if (!current_->isInGlobal()) {
        return;
    }

    if (!node->init) throw GlobalError(node->line, "Uninitialized global");
    if (!isScalarType(node->init->type.baseType)) throw GlobalError(node->line, "Non-scalar global variables are illegal");
    if (node->qualifier == "var") throw GlobalError(node->line, "'var' qualifier in global scope");
}