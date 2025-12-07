#include "CompileTimeExceptions.h"
#include "MLIRgen.h"
#include "Types.h"

VarInfo MLIRGen::popValue() {
    if (v_stack_.empty()) {
        throw std::runtime_error("MLIRGen internal error: value stack underflow.");
    }
    VarInfo v = std::move(v_stack_.back());
    v_stack_.pop_back();
    return v;
}

void MLIRGen::pushValue(VarInfo& value) {
    v_stack_.push_back(value);
}
mlir::Value MLIRGen::getSSAValue(const VarInfo &v){
    if(!v.value.getType().isa<mlir::MemRefType>()) return v.value; //already an SSA value
    // if a memref type - convert to a SSA type
    return builder_.create<mlir::memref::LoadOp>(loc_, v.value, mlir::ValueRange{}).getResult();
}

void MLIRGen::syncRuntimeDims(VarInfo* var) {
    if (!var) return;

    BaseType bt = var->type.baseType;
    if (bt != BaseType::ARRAY && bt != BaseType::VECTOR && bt != BaseType::MATRIX) {
        // For non array-like values we do not track runtimeDims.
        // Ensure we never carry more than 2 entries to keep invariants simple.
        if (var->runtimeDims.size() > 2) {
            var->runtimeDims.resize(2);
        }
        return;
    }

    auto &staticDims = var->type.dims;
    auto &rtDims = var->runtimeDims;

    // Hard cap: we never want to carry more than 2 runtime dimensions.
    if (rtDims.size() > 2) {
        rtDims.resize(2);
    }

    // Helper lambdas to pick a dimension from runtime vs static info.
    auto pickDim = [](int rt, int st) -> int {
        // Prefer a positive runtime value if available,
        // otherwise fall back to a positive static dimension.
        if (rt > 0) return rt;
        if (st > 0) return st;
        // Unknown / inferred at runtime.
        return -1;
    };

    if (bt == BaseType::ARRAY || bt == BaseType::VECTOR) {
        // For true vectors and 1D arrays we normalise to rank-1.
        // For ARRAY types that are explicitly 2D at the type level
        // (e.g., used to represent nested composites), preserve a 2D
        // shape instead of collapsing.
        if (bt == BaseType::ARRAY && staticDims.size() >= 2) {
            int rt0 = rtDims.size() > 0 ? rtDims[0] : -1;
            int rt1 = rtDims.size() > 1 ? rtDims[1] : -1;
            int st0 = staticDims[0];
            int st1 = staticDims[1];

            int final0 = pickDim(rt0, st0);
            int final1 = pickDim(rt1, st1);
            rtDims.clear();
            rtDims.push_back(final0);
            rtDims.push_back(final1);
        } else {
            // Canonical rank-1 shape. Collapse any higher-rank info into a
            // single effective length using either runtime or static info.
            int rt0 = !rtDims.empty() ? rtDims[0] : -1;
            int st0 = !staticDims.empty() ? staticDims[0] : -1;

            if (rtDims.size() > 1) {
                long long prod = 1;
                bool hasPositive = false;
                for (int d : rtDims) {
                    if (d > 0) {
                        hasPositive = true;
                        prod *= d;
                    }
                }
                if (hasPositive) {
                    rt0 = static_cast<int>(prod);
                }
            }

            int finalLen = pickDim(rt0, st0);
            rtDims.clear();
            rtDims.push_back(finalLen);
        }
    } else if (bt == BaseType::MATRIX) {
        // Canonical rank-2 shape. Ensure exactly two runtime dims, seeded from
        // runtime where available and otherwise from static dims.
        int rt0 = rtDims.size() > 0 ? rtDims[0] : -1;
        int rt1 = rtDims.size() > 1 ? rtDims[1] : -1;
        int st0 = staticDims.size() > 0 ? staticDims[0] : -1;
        int st1 = staticDims.size() > 1 ? staticDims[1] : -1;

        int final0 = pickDim(rt0, st0);
        int final1 = pickDim(rt1, st1);
        rtDims.clear();
        rtDims.push_back(final0);
        rtDims.push_back(final1);
    }
}

void MLIRGen::syncRuntimeDims(CompleteType& promotedType, const VarInfo& lhs, const VarInfo& rhs) {
    // Only applicable for array-like types
    if (promotedType.baseType != BaseType::ARRAY && 
        promotedType.baseType != BaseType::VECTOR && 
        promotedType.baseType != BaseType::MATRIX) {
        return;
    }

    // Check if we have wildcards
    bool hasWildcard = false;
    for (int d : promotedType.dims) {
        if (d < 0) { hasWildcard = true; break; }
    }
    if (!hasWildcard && !promotedType.dims.empty()) return; // All static dims known

    // Helper to copy from source if compatible
    auto copyDims = [&](const VarInfo& src) -> bool {
        if (src.runtimeDims.empty()) return false;
        
        int promoRank = promotedType.baseType == BaseType::MATRIX ? 2 : 1;
        if (promotedType.dims.size() > 0) promoRank = promotedType.dims.size();
        
        int srcRank = src.runtimeDims.size();
        
        if (srcRank == promoRank) {
             // Ensure source dims are valid
             for (int d : src.runtimeDims) if (d <= 0) return false;
             
             promotedType.dims = src.runtimeDims;
             return true;
        }
        return false;
    };

    if (copyDims(lhs)) return;
    copyDims(rhs);
}
// Declarations / Globals helpers
mlir::Type MLIRGen::getLLVMType(const CompleteType& type) {
    switch (type.baseType) {
        case BaseType::BOOL: return builder_.getI1Type();
        case BaseType::CHARACTER: return builder_.getI8Type();
        case BaseType::INTEGER: return builder_.getI32Type();
        case BaseType::REAL: return builder_.getF32Type();
        case BaseType::TUPLE:
        case BaseType::STRUCT: {
            if (type.subTypes.empty()) {
                throw std::runtime_error("getLLVMType: empty aggregate type is invalid");
            }
            llvm::SmallVector<mlir::Type, 4> elemTys;
            elemTys.reserve(type.subTypes.size());
            for (const auto &st : type.subTypes) {
                elemTys.push_back(getLLVMType(st));
            }
            return mlir::LLVM::LLVMStructType::getLiteral(&context_, elemTys);
        }
        case BaseType::ARRAY: {
            throw std::runtime_error("getLLVMType: Arrays should not be called with this helper.");
        }
        case BaseType::VECTOR: {
            throw std::runtime_error("getLLVMType: Vectors should not be called with this helper.");
        }
        default:
            throw std::runtime_error("getLLVMType: Unsupported type");
    }
}
// Create a VarInfo that contains an allocated memref with the compile-time
// constant stored. Supports scalar types (int, real, bool, char). Throws on
// unsupported types.
VarInfo MLIRGen::createLiteralFromConstant(const ConstantValue &cv, const CompleteType &type, int line) {
    VarInfo lit(type);
    // allocate a literal container (memref) and mark it const
    allocaLiteral(&lit, line);

    switch (cv.type.baseType) {
        case BaseType::INTEGER: {
            auto i32 = builder_.getI32Type();
            int64_t v = std::get<int64_t>(cv.value);
            auto c = builder_.create<mlir::arith::ConstantOp>(loc_, i32, builder_.getIntegerAttr(i32, static_cast<int64_t>(v)));
            builder_.create<mlir::memref::StoreOp>(loc_, c, lit.value, mlir::ValueRange{});
            break;
        }
        case BaseType::REAL: {
            auto f32 = builder_.getF32Type();
            double dv = std::get<double>(cv.value);
            auto c = builder_.create<mlir::arith::ConstantOp>(loc_, f32, builder_.getFloatAttr(f32, static_cast<float>(dv)));
            builder_.create<mlir::memref::StoreOp>(loc_, c, lit.value, mlir::ValueRange{});
            break;
        }
        case BaseType::BOOL: {
            auto i1 = builder_.getI1Type();
            bool bv = std::get<bool>(cv.value);
            auto c = builder_.create<mlir::arith::ConstantOp>(loc_, i1, builder_.getIntegerAttr(i1, bv ? 1 : 0));
            builder_.create<mlir::memref::StoreOp>(loc_, c, lit.value, mlir::ValueRange{});
            break;
        }
        case BaseType::CHARACTER: {
            auto i8 = builder_.getI8Type();
            char ch = std::get<char>(cv.value);
            auto c = builder_.create<mlir::arith::ConstantOp>(loc_, i8, builder_.getIntegerAttr(i8, static_cast<int>(ch)));
            builder_.create<mlir::memref::StoreOp>(loc_, c, lit.value, mlir::ValueRange{});
            break;
        }
        case BaseType::MATRIX: {
            // Matrix constant is a vector<ConstantValue> where each element is a row (ARRAY of scalars)
            if (!std::holds_alternative<std::vector<ConstantValue>>(cv.value)) {
                throw std::runtime_error("createLiteralFromConstant: MATRIX constant has invalid structure");
            }
            
            const auto &rows = std::get<std::vector<ConstantValue>>(cv.value);
            if (rows.empty()) {
                throw std::runtime_error("createLiteralFromConstant: empty matrix constant");
            }
            
            // Allocate 2D memref
            if (!lit.value) {
                allocaLiteral(&lit, line);
            }
            
            // Store each element: matrix[i][j] = rows[i].elements[j]
            auto idxTy = builder_.getIndexType();
            for (size_t i = 0; i < rows.size(); ++i) {
                if (!std::holds_alternative<std::vector<ConstantValue>>(rows[i].value)) {
                    throw std::runtime_error("createLiteralFromConstant: matrix row is not a vector");
                }
                const auto &row = std::get<std::vector<ConstantValue>>(rows[i].value);
                
                for (size_t j = 0; j < row.size(); ++j) {
                    const auto &elem = row[j];
                    mlir::Value elemVal;
                    
                    if (elem.type.baseType == BaseType::INTEGER) {
                        auto i32 = builder_.getI32Type();
                        int64_t v = std::get<int64_t>(elem.value);
                        elemVal = builder_.create<mlir::arith::ConstantOp>(loc_, i32, builder_.getIntegerAttr(i32, v));
                    } else if (elem.type.baseType == BaseType::REAL) {
                        auto f32 = builder_.getF32Type();
                        double dv = std::get<double>(elem.value);
                        elemVal = builder_.create<mlir::arith::ConstantOp>(loc_, f32, builder_.getFloatAttr(f32, static_cast<float>(dv)));
                    } else {
                        throw std::runtime_error("createLiteralFromConstant: unsupported matrix element type");
                    }
                    
                    mlir::Value idxI = builder_.create<mlir::arith::ConstantOp>(
                        loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(i)));
                    mlir::Value idxJ = builder_.create<mlir::arith::ConstantOp>(
                        loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(j)));
                    
                    builder_.create<mlir::memref::StoreOp>(loc_, elemVal, lit.value, mlir::ValueRange{idxI, idxJ});
                }
            }
            break;
        }
        default:
            throw std::runtime_error("createLiteralFromConstant: unsupported constant type");
    }

    return lit;
}

mlir::Attribute MLIRGen::extractConstantValue(std::shared_ptr<ExprNode> expr, const CompleteType& targetType) {
    if (!expr) throw std::runtime_error("FATAL: no initializer for global.");
    if (auto tn = std::dynamic_pointer_cast<TrueNode>(expr)) {
        return builder_.getIntegerAttr(builder_.getI1Type(), 1);
    }
    if (auto fn = std::dynamic_pointer_cast<FalseNode>(expr)) {
        return builder_.getIntegerAttr(builder_.getI1Type(), 0);
    }
    if (auto cn = std::dynamic_pointer_cast<CharNode>(expr)) {
        return builder_.getIntegerAttr(builder_.getI8Type(), static_cast<int>(cn->value));
    }
    if (auto in = std::dynamic_pointer_cast<IntNode>(expr)) {
        return builder_.getIntegerAttr(builder_.getI32Type(), in->value);
    }
    if (auto rn = std::dynamic_pointer_cast<RealNode>(expr)) {
        return builder_.getFloatAttr(builder_.getF32Type(), rn->value);
    }
    // Tuple literals are handled element-wise for tuple globals; this helper
    // remains scalar-only.
    (void)targetType;
    return nullptr;
}

mlir::Value MLIRGen::createGlobalVariable(const std::string& name, const CompleteType& type, bool isConst, mlir::Attribute initValue) {
    mlir::Type globalTy = getLLVMType(type);
    auto* moduleBuilder = backend_.getBuilder().get();
    auto savedIP = moduleBuilder->saveInsertionPoint();
    moduleBuilder->setInsertionPointToStart(module_.getBody());
    moduleBuilder->create<mlir::LLVM::GlobalOp>(
        loc_, globalTy, isConst, mlir::LLVM::Linkage::Internal, name, initValue, 0);
    moduleBuilder->restoreInsertionPoint(savedIP);
    return nullptr;
}

void MLIRGen::assignTo(VarInfo* literal, VarInfo* variable, int line) {
    // Tuple assignment: element-wise store with implicit scalar promotions
    if (variable->type.baseType == BaseType::TUPLE) {
        if (literal->type.baseType != BaseType::TUPLE) {
            throw AssignError(line, "Cannot assign non-tuple to tuple variable '");
        }
        // Ensure destination tuple storage exists
        if (!variable->value) {
            allocaVar(variable, line);
        }
        if (!literal->value) {
            allocaVar(literal, line);
        }
        if (literal->type.baseType != BaseType::TUPLE ||
            literal->type.subTypes.size() != variable->type.subTypes.size()) {
            throw AssignError(line, "Tuple arity mismatch in assignment.");
        }

        // Use castType to perform any element-wise casts, then copy struct.
        VarInfo converted = castType(literal, &variable->type, line);
        mlir::Type structTy = getLLVMType(variable->type);
        mlir::Value srcStruct = builder_.create<mlir::LLVM::LoadOp>(
            loc_, structTy, converted.value);
        builder_.create<mlir::LLVM::StoreOp>(
            loc_, srcStruct, variable->value);
        return;
    } else if (variable->type.baseType == BaseType::ARRAY){
        assignToArray(literal, variable, line);
        return;
    } else if (variable->type.baseType == BaseType::VECTOR) {
        assignToVector(literal, variable, line);
        return;
    } else if (variable->type.baseType == BaseType::STRUCT) {
        // Ensure destination and source storage exist
        if (!variable->value) {
            allocaVar(variable, line);
        }
        if (!literal->value) {
            allocaVar(literal, line);
        }

        // Types should already be checked by semantic analysis
        // Perform whole-struct load/store in LLVM dialect.
        mlir::Type structTy = getLLVMType(variable->type);
        mlir::Value srcStruct =
            builder_.create<mlir::LLVM::LoadOp>(loc_, structTy, literal->value);
        builder_.create<mlir::LLVM::StoreOp>(loc_, srcStruct, variable->value);
        return;
    } else if ((variable->type.baseType == BaseType::ARRAY || variable->type.baseType == BaseType::MATRIX) &&
               (literal->type.baseType == BaseType::ARRAY || literal->type.baseType == BaseType::MATRIX)) {
        // Array/Matrix assignment
        assignToArray(literal, variable, line);
        return;
    } else if (isScalarType(variable->type.baseType) && isScalarType(literal->type.baseType)) {

        // Scalar assignment
        // ensure var has a memref allocated
        if (!variable->value) {
            allocaVar(variable, line);
        }

        VarInfo promoted = promoteType(literal, &variable->type, line); // handle type promotions + errors

        // Normalize promoted value to SSA (load memref if needed)
        mlir::Value loadedVal = getSSAValue(promoted);
        builder_.create<mlir::memref::StoreOp>(
            loc_, loadedVal, variable->value, mlir::ValueRange{}
        );
    } else {
        throw std::runtime_error("MLIRGen::assignTo: unsupported variable type");
    }


}

void MLIRGen::assignToArray(VarInfo* rhs, VarInfo* lhs, int line) {
    if ((rhs->type.baseType != BaseType::ARRAY && rhs->type.baseType != BaseType::VECTOR && rhs->type.baseType != BaseType::MATRIX) ||
        (lhs->type.baseType != BaseType::ARRAY && lhs->type.baseType != BaseType::VECTOR && lhs->type.baseType != BaseType::MATRIX)) {
        throw AssignError(line, "Cannot assign type " + toString(rhs->type) + " to " + toString(lhs->type));
    }

    // Ensure both storages exist
    if (!lhs->value) allocaVar(lhs, line);
    if (!rhs->value) allocaVar(rhs, line);

    syncRuntimeDims(lhs);
    syncRuntimeDims(rhs);

    // Ensure same dimension count
    if (lhs->runtimeDims.size() != rhs->runtimeDims.size()) {
        throw SizeError(line, "Mismatched lhs and rhs dimensions of " + std::to_string(lhs->runtimeDims.size()) + " and " + std::to_string(rhs->runtimeDims.size()));
    }

    // Handle lhs inferred dimensions ('*') using rhs dimensions. SizeError if dimension cannot be inferred.
    for (size_t i = 0; i < lhs->runtimeDims.size(); ++i) {
        int &lhsDim = lhs->runtimeDims[i];
        int rhsDim = rhs->runtimeDims[i];

        if (lhsDim < 0) {
            if (rhsDim < 0) {
                throw SizeError(line, "Missing runtime dimensions for array/vector assignment");
            }

            // Save lhs dims from rhs
            lhsDim = rhsDim;
            if (i < lhs->type.dims.size()) {
                lhs->type.dims[i] = rhsDim;
            }
        }
    }

    // Handle 2D array/matrix assignment
    if (lhs->runtimeDims.size() == 2) {
        int lhsRows = lhs->runtimeDims[0];
        int lhsCols = lhs->runtimeDims[1];
        int rhsRows = rhs->runtimeDims[0];
        int rhsCols = rhs->runtimeDims[1];

        if (lhsRows != rhsRows || lhsCols != rhsCols) {
            throw SizeError(line, "Mismatched matrix dimensions");
        }

        auto idxTy = builder_.getIndexType();
        for (int i = 0; i < lhsRows; ++i) {
            for (int j = 0; j < lhsCols; ++j) {
                mlir::Value idxI = builder_.create<mlir::arith::ConstantOp>(
                    loc_, idxTy, builder_.getIntegerAttr(idxTy, i));
                mlir::Value idxJ = builder_.create<mlir::arith::ConstantOp>(
                    loc_, idxTy, builder_.getIntegerAttr(idxTy, j));
                
                mlir::Value srcVal = builder_.create<mlir::memref::LoadOp>(
                    loc_, rhs->value, mlir::ValueRange{idxI, idxJ});
                builder_.create<mlir::memref::StoreOp>(
                    loc_, srcVal, lhs->value, mlir::ValueRange{idxI, idxJ});
            }
        }
        return;
    }

    // Handle 1D array/vector assignment
    if (lhs->runtimeDims.empty() || rhs->runtimeDims.empty()) {
        throw SizeError(line, "Missing runtime dimensions for array/vector assignment");
    }
    int lhsLen = lhs->runtimeDims[0];
    int rhsLen = rhs->runtimeDims[0];

    // Enforce same length assignment only, only exception being vectors.
    // A smaller vector results in 0-padding, a larger vector results in slicing.
    if (lhsLen != rhsLen && rhs->type.baseType != BaseType::VECTOR) {
        throw SizeError(line, "Mismatched lhs (array) and rhs lengths of " +
                                  std::to_string(lhsLen) + " and " + std::to_string(rhsLen));
    }

    size_t n = lhsLen;
    auto idxTy = builder_.getIndexType();
    for(size_t t=0; t<n;++t){
            // index constant
        mlir::Value idx = builder_.create<mlir::arith::ConstantOp>(
            loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(t)));

        // If the rhs is a smaller vector, we zero pad it with the lhs' element type.
        if (rhs->type.baseType == BaseType::VECTOR && t >= rhsLen) {
            BaseType elemBaseType = lhs->type.subTypes[0].baseType;
            mlir::Type elemTy = getLLVMType(lhs->type.subTypes[0]);
            mlir::Value zero;

            if (elemBaseType == BaseType::INTEGER || elemBaseType == BaseType::BOOL || elemBaseType == BaseType::CHARACTER) {
                zero = builder_.create<mlir::arith::ConstantOp>(loc_, elemTy, builder_.getIntegerAttr(elemTy, 0));
            } else if (elemBaseType == BaseType::REAL) {
                zero = builder_.create<mlir::arith::ConstantOp>(
                    loc_, elemTy, builder_.getFloatAttr(elemTy, 0.0));
            } else {
                throw std::runtime_error("assignToArray: unsupported element type for zero padding");
            }
            builder_.create<mlir::memref::StoreOp>(loc_, zero, lhs->value, mlir::ValueRange{idx});
            continue;
        }

        // load source element
        mlir::Value srcElemVal = builder_.create<mlir::memref::LoadOp>(loc_, rhs->value, mlir::ValueRange{idx});

        // build a temp VarInfo for the source element
        if (rhs->type.subTypes.size() != 1) {
            throw std::runtime_error("varHelper::assignTo: array rhs type must have exactly one element subtype");
        }
        CompleteType srcElemCT = rhs->type.subTypes[0];
        VarInfo srcElemVar(srcElemCT);
        srcElemVar.value = srcElemVal;
        srcElemVar.isLValue = false;

        // destination element type
        if (lhs->type.subTypes.size() != 1) {
            throw std::runtime_error("varHelper::assignTo: array lhs type must have exactly one element subtype");
        }
        CompleteType dstElemCT = lhs->type.subTypes[0];

        // promote/cast
        VarInfo promoted = promoteType(&srcElemVar, &dstElemCT, line);
        mlir::Value storeVal = getSSAValue(promoted);

        // store into destination
        builder_.create<mlir::memref::StoreOp>(loc_, storeVal, lhs->value, mlir::ValueRange{idx});
    }
}

void MLIRGen::assignToVector(VarInfo* literal, VarInfo* variable, int line) {
    if (literal->type.baseType != BaseType::VECTOR && literal->type.baseType != BaseType::ARRAY) {
        throw TypeError(line, "MLIRGen::assignToVector: Literal of type '" + toString(literal->type) + "' being assigned to vector variable");
        // TODO: Implement int -> vector
    }
    // Ensure both storages exist
    if (!variable->value) allocaVar(variable, line);
    if (!literal->value) allocaVar(literal, line);

    // Seed runtime dimensions from types where missing
    if (literal->runtimeDims.empty()) {
        literal->runtimeDims = literal->type.dims;
    }

    // Determine new runtime length from literal
    if (literal->runtimeDims.empty() || literal->runtimeDims[0] < 0) {
        throw std::runtime_error("MLIRGen::assignToVector: invalid literal runtime dimensions for vector assignment");
    }
    int newLen = literal->runtimeDims[0];
    if (newLen < 0) {
        throw std::runtime_error("MLIRGen::assignToVector: Runtime len of " + std::to_string(newLen) + " is invalid");
    }
    variable->runtimeDims = {newLen};

    // resize the vector to the rhs
    mlir::Value newVector = allocaVector(newLen, variable);
    variable->value = newVector;

    // Copy over elements to the new vector
    auto idxTy = builder_.getIndexType();
    for (int t = 0; t < newLen; ++t) {
        // index constant
        mlir::Value idx = builder_.create<mlir::arith::ConstantOp>(
            loc_, idxTy, builder_.getIntegerAttr(idxTy, static_cast<int64_t>(t)));

        // load source element
        mlir::Value srcElemVal = builder_.create<mlir::memref::LoadOp>(loc_, literal->value, mlir::ValueRange{idx});

        // build a temp VarInfo for the source element
        if (literal->type.subTypes.size() != 1) {
            throw std::runtime_error("varHelper::assignTo: array literal type must have exactly one element subtype");
        }
        CompleteType srcElemCT = literal->type.subTypes[0];
        VarInfo srcElemVar(srcElemCT);
        srcElemVar.value = srcElemVal;
        srcElemVar.isLValue = false;

        // destination element type
        if (variable->type.subTypes.size() != 1) {
            throw std::runtime_error("varHelper::assignTo: array variable type must have exactly one element subtype");
        }
        CompleteType dstElemCT = variable->type.subTypes[0];

        // promote/cast
        VarInfo promoted = promoteType(&srcElemVar, &dstElemCT, line);
        mlir::Value storeVal = getSSAValue(promoted);

        // store into destination
        builder_.create<mlir::memref::StoreOp>(loc_, storeVal, variable->value, mlir::ValueRange{idx});
    }
}

void MLIRGen::allocaLiteral(VarInfo* varInfo, int line) {
    varInfo->isConst = true;
    allocaVar(varInfo, line);
}

bool MLIRGen::tryEmitConstantForNode(ExprNode* node) {
    if (!node) return false;
    if (!node->constant.has_value()) return false;
    try {
        const ConstantValue &cv = node->constant.value();
        // For scalar constants, emit SSA arith::ConstantOp instead of
        // allocating a memref and storing into it. Tuples and complex
        // constants still use the memref literal path.
        switch (cv.type.baseType) {
            case BaseType::INTEGER: {
                auto i32 = builder_.getI32Type();
                int64_t v = std::get<int64_t>(cv.value);
                auto c = builder_.create<mlir::arith::ConstantOp>(loc_, i32, builder_.getIntegerAttr(i32, v));
                VarInfo out{CompleteType(BaseType::INTEGER)};
                out.value = c.getResult();
                out.isLValue = false;
                pushValue(out);
                return true;
            }
            case BaseType::REAL: {
                auto f32 = builder_.getF32Type();
                double dv = std::get<double>(cv.value);
                auto c = builder_.create<mlir::arith::ConstantOp>(loc_, f32, builder_.getFloatAttr(f32, static_cast<float>(dv)));
                VarInfo out{CompleteType(BaseType::REAL)};
                out.value = c.getResult();
                out.isLValue = false;
                pushValue(out);
                return true;
            }
            case BaseType::BOOL: {
                auto i1 = builder_.getI1Type();
                bool bv = std::get<bool>(cv.value);
                auto c = builder_.create<mlir::arith::ConstantOp>(loc_, i1, builder_.getIntegerAttr(i1, bv ? 1 : 0));
                VarInfo out{CompleteType(BaseType::BOOL)};
                out.value = c.getResult();
                out.isLValue = false;
                pushValue(out);
                return true;
            }
            case BaseType::CHARACTER: {
                auto i8 = builder_.getI8Type();
                char ch = std::get<char>(cv.value);
                auto c = builder_.create<mlir::arith::ConstantOp>(loc_, i8, builder_.getIntegerAttr(i8, static_cast<int>(ch)));
                VarInfo out{CompleteType(BaseType::CHARACTER)};
                out.value = c.getResult();
                out.isLValue = false;
                pushValue(out);
                return true;
            }
            default: {
                // Fallback: use existing memref literal path for tuples and unsupported types
                VarInfo lit = createLiteralFromConstant(node->constant.value(), node->type, node->line);
                pushValue(lit);
                return true;
            }
        }
    } catch (...) {
        // unsupported constant type or codegen error; fall back to normal lowering
        return false;
    }
}

mlir::func::FuncOp MLIRGen::getCurrentEnclosingFunction() {
    mlir::Block *block = builder_.getBlock();
    if (!block) block = builder_.getInsertionBlock();
    if (!block) {
        throw std::runtime_error("allocaVar: builder has no current block");
    }

    mlir::Operation *op = block->getParentOp();
    while (op) {
        if (auto f = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
            return f;
        }
        op = op->getParentOp();
    }

    throw std::runtime_error("allocaVar: could not find parent function for allocation");
}

void MLIRGen::allocaVar(VarInfo* varInfo, int line) {
    mlir::func::FuncOp funcOp = getCurrentEnclosingFunction();
    mlir::Block &entry = funcOp.front();
    mlir::OpBuilder entryBuilder(&entry, entry.begin());

    switch (varInfo->type.baseType) {
        case BaseType::BOOL:
            varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI1Type()));
            break;
        case BaseType::CHARACTER:
            varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI8Type()));
            break;
        case BaseType::INTEGER:
            varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI32Type()));
            break;
        case BaseType::REAL:
            varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getF32Type()));
            break;

        case BaseType::TUPLE:
        case BaseType::STRUCT: {
            if (varInfo->type.subTypes.size() < 1) {
                throw SizeError(line, "Error: aggregate must have at least 1 element.");
            }
            mlir::Type structTy = getLLVMType(varInfo->type);
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
            auto i64Ty = builder_.getI64Type();
            auto oneAttr = builder_.getIntegerAttr(i64Ty, 1);
            mlir::Value one = entryBuilder.create<mlir::arith::ConstantOp>(loc_, i64Ty, oneAttr);
            varInfo->value = entryBuilder.create<mlir::LLVM::AllocaOp>(loc_, ptrTy, structTy, one, 0u);
            break;
        }
        case BaseType::ARRAY:{ 
            mlir::Type elemTy = getLLVMType(varInfo->type.subTypes[0]);

            if (varInfo->type.dims.size() == 1 && varInfo->type.dims[0] >= 0) {
                int64_t n = varInfo->type.dims[0];
                auto memTy = mlir::MemRefType::get({n}, elemTy);
                varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                    loc_, memTy, mlir::ValueRange{}, mlir::ValueRange{}, builder_.getIntegerAttr(builder_.getI64Type(), 16));
            } else if (varInfo->type.dims.size() == 2 && varInfo->type.dims[0] >= 0 && varInfo->type.dims[1] >= 0) {
                // 2D array (matrix)
                int64_t rows = varInfo->type.dims[0];
                int64_t cols = varInfo->type.dims[1];
                auto memTy = mlir::MemRefType::get({rows, cols}, elemTy);
                varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                    loc_, memTy, mlir::ValueRange{}, mlir::ValueRange{}, builder_.getIntegerAttr(builder_.getI64Type(), 16));
            } else {
                throw std::runtime_error("allocaVar: unsupported array shape with dimension " + std::to_string(varInfo->type.dims.size()) + " for variable '" +
                                         (varInfo->identifier.empty() ? std::string("<unknown>") : varInfo->identifier) + "'");
            }
            break;
        }

        case BaseType::MATRIX: {
            // Matrix is represented as 2D memref
            if (varInfo->type.subTypes.empty()) {
                throw std::runtime_error("allocaVar: MATRIX type has no element subtype for variable '" +
                                         (varInfo->identifier.empty() ? std::string("<unknown>") : varInfo->identifier) + "'");
            }
            mlir::Type elemTy = getLLVMType(varInfo->type.subTypes[0]);
            if (varInfo->type.dims.size() != 2) {
                throw std::runtime_error("allocaVar: MATRIX requires 2 dimensions but got " + 
                                         std::to_string(varInfo->type.dims.size()) + " for variable '" +
                                         (varInfo->identifier.empty() ? std::string("<unknown>") : varInfo->identifier) + "'");
            }
            if (varInfo->type.dims[0] < 0 || varInfo->type.dims[1] < 0) {
                throw std::runtime_error("allocaVar: MATRIX dimensions cannot be inferred (dims: " + 
                                         std::to_string(varInfo->type.dims[0]) + "," + std::to_string(varInfo->type.dims[1]) + 
                                         ") for variable '" +
                                         (varInfo->identifier.empty() ? std::string("<unknown>") : varInfo->identifier) + "'");
            }
            
            int64_t rows = varInfo->type.dims[0];
            int64_t cols = varInfo->type.dims[1];
            auto memTy = mlir::MemRefType::get({rows, cols}, elemTy);
            varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                loc_, memTy, mlir::ValueRange{}, mlir::ValueRange{}, builder_.getIntegerAttr(builder_.getI64Type(), 16));
            break;
        }

        case BaseType::VECTOR: {
            if (varInfo->type.subTypes.size() != 1) {
                throw std::runtime_error("allocaVar: Vector with size != 1 found");
            }

            mlir::Type elemTy = getLLVMType(varInfo->type.subTypes[0]);
            auto memTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);


            auto idxTy = builder_.getIndexType();
            mlir::Value zeroLen = entryBuilder.create<mlir::arith::ConstantOp>(
                loc_, idxTy, builder_.getIntegerAttr(idxTy, 0));

            varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                loc_,
                memTy,
                mlir::ValueRange{zeroLen}, // one dynamic size operand
                mlir::ValueRange{},
                builder_.getIntegerAttr(builder_.getI64Type(), 16)
            );
            break;
        }

        default: {
            std::string varName = varInfo->identifier.empty() ? "<temporary>" : varInfo->identifier;
            throw std::runtime_error("allocaVar: unsupported type " +
                                     std::to_string(static_cast<int>(varInfo->type.baseType)) +
                                     " for variable '" + varName + "'");
        }
    }

    syncRuntimeDims(varInfo);
}

mlir::Value MLIRGen::allocaVector(int len, VarInfo *varInfo) {
    // get front of block to ensure alloca's are at top of block
    mlir::func::FuncOp funcOp = getCurrentEnclosingFunction();
    mlir::Block &entry = funcOp.front();
    mlir::OpBuilder entryBuilder(&entry, entry.begin());


    mlir::Type elemTy = getLLVMType(varInfo->type.subTypes[0]);
    auto memTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
    mlir::Value arrayLen = entryBuilder.create<mlir::arith::ConstantOp>(
        loc_, builder_.getIndexType(), builder_.getIntegerAttr(builder_.getIndexType(), len));
    
    mlir::Value alloca;
    BaseType bt = varInfo->type.subTypes[0].baseType;
    if (bt == BaseType::CHARACTER || bt == BaseType::BOOL) {
        alloca = entryBuilder.create<mlir::memref::AllocOp>(
            loc_,
            memTy,
            mlir::ValueRange{arrayLen},
            mlir::ValueRange{});
    } else {
        alloca = entryBuilder.create<mlir::memref::AllocaOp>(
            loc_,
            memTy,
            mlir::ValueRange{arrayLen},
            mlir::ValueRange{});
    }
    varInfo->value = alloca;
    return alloca;
}

void MLIRGen::zeroInitializeVar(VarInfo* var) {
    if (!var->value) return;

    mlir::Value zeroVal;
    
    // Determine scalar zero value based on type
    switch(var->type.baseType) {
        case BaseType::INTEGER: {
            auto t = builder_.getI32Type();
            zeroVal = builder_.create<mlir::arith::ConstantOp>(loc_, t, builder_.getIntegerAttr(t, 0));
            break;
        }
        case BaseType::REAL: {
            auto t = builder_.getF32Type();
            zeroVal = builder_.create<mlir::arith::ConstantOp>(loc_, t, builder_.getFloatAttr(t, 0.0));
            break;
        }
        case BaseType::BOOL: {
            auto t = builder_.getI1Type();
            zeroVal = builder_.create<mlir::arith::ConstantOp>(loc_, t, builder_.getIntegerAttr(t, 0));
            break;
        }
        case BaseType::CHARACTER: {
            auto t = builder_.getI8Type();
            zeroVal = builder_.create<mlir::arith::ConstantOp>(loc_, t, builder_.getIntegerAttr(t, 0));
            break;
        }
        default:
            // Tuple zeroing logic is handled by iterating elements in visitor, not here
            return;
    }

    if (zeroVal) {
        if (var->value.getType().isa<mlir::MemRefType>()) {
            builder_.create<mlir::memref::StoreOp>(loc_, zeroVal, var->value, mlir::ValueRange{});
        } else {
            // Assume LLVM pointer (e.g. tuple element via GEP)
            builder_.create<mlir::LLVM::StoreOp>(loc_, zeroVal, var->value);
        }
    }
}
