#include "CompileTimeExceptions.h"
#include "MLIRgen.h"
#include "Types.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/Support/raw_ostream.h" // For llvm::errs()
#include "llvm/ADT/ArrayRef.h"
#include <iostream> // For std::cerr
#include "mlir/Dialect/SCF/IR/SCF.h" // Needed for scf::ForOp

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
    if(v.value.getType().isa<mlir::MemRefType>()) {
        return builder_.create<mlir::memref::LoadOp>(loc_, v.value, mlir::ValueRange{}).getResult();
    }
    if(v.value.getType().isa<mlir::LLVM::LLVMPointerType>()) {
        mlir::Type valTy = getLLVMType(v.type);
        return builder_.create<mlir::LLVM::LoadOp>(loc_, valTy, v.value);
    }
    return v.value; //already an SSA value
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
        // Prefer a valid runtime value if available,
        // otherwise fall back to a valid static dimension.
        if (rt >= 0) return rt;
        if (st >= 0) return st;
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
        case BaseType::STRING: {
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
            auto i64Ty = builder_.getI64Type();
            llvm::SmallVector<mlir::Type, 2> fields{ptrTy, i64Ty};
            return mlir::LLVM::LLVMStructType::getLiteral(&context_, fields);
        }
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
            // Descriptor support for array: { ptr, len } or { ptr, rows, cols }
            if (type.subTypes.empty()) {
                throw std::runtime_error("getLLVMType: ARRAY type must have at least one element subtype");
            }
            // Validate element type is representable in LLVM (don't need the value)
            getLLVMType(type.subTypes[0]);
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
            auto i64Ty = builder_.getI64Type();
            if (type.dims.empty() || type.dims.size() == 1) {
                llvm::SmallVector<mlir::Type, 2> fields{ptrTy, i64Ty};
                return mlir::LLVM::LLVMStructType::getLiteral(&context_, fields);
            } else if (type.dims.size() == 2) {
                llvm::SmallVector<mlir::Type, 3> fields{ptrTy, i64Ty, i64Ty};
                return mlir::LLVM::LLVMStructType::getLiteral(&context_, fields);
            } else {
                throw std::runtime_error("getLLVMType: ARRAY descriptor does not support rank > 2 (got dims.size() = " + std::to_string(type.dims.size()) + ")");
            }
        }
        case BaseType::VECTOR: {
            // Descriptor support for vector: { ptr, len }
            if (type.subTypes.size() != 1) {
                throw std::runtime_error("getLLVMType: VECTOR type must have exactly one element subtype");
            }
            getLLVMType(type.subTypes[0]);
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
            auto i64Ty = builder_.getI64Type();
            llvm::SmallVector<mlir::Type, 2> fields{ptrTy, i64Ty};
            return mlir::LLVM::LLVMStructType::getLiteral(&context_, fields);
        }
        case BaseType::MATRIX: {
            // Descriptor support for matrix: { ptr, rows, cols }
            if (type.subTypes.size() != 1) {
                throw std::runtime_error("getLLVMType: MATRIX type must have exactly one element subtype");
            }
            getLLVMType(type.subTypes[0]);
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
            auto i64Ty = builder_.getI64Type();
            llvm::SmallVector<mlir::Type, 3> fields{ptrTy, i64Ty, i64Ty};
            return mlir::LLVM::LLVMStructType::getLiteral(&context_, fields);
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

mlir::Value MLIRGen::accessElement(VarInfo* var, mlir::ValueRange indices, mlir::Value storeVal) {
    if (var->value.getType().isa<mlir::MemRefType>()) {
        if (storeVal) {
            builder_.create<mlir::memref::StoreOp>(loc_, storeVal, var->value, indices);
            return nullptr;
        } else {
            return builder_.create<mlir::memref::LoadOp>(loc_, var->value, indices);
        }
    }
    
    mlir::Value descriptor = var->value;
    // If it's a pointer to descriptor (e.g. local mutable vector/string), load it first
    if (var->value.getType().isa<mlir::LLVM::LLVMPointerType>()) {
        // Load the descriptor
        mlir::Type descTy = getLLVMType(var->type);
        descriptor = builder_.create<mlir::LLVM::LoadOp>(loc_, descTy, var->value);
    }
    
    if (descriptor.getType().isa<mlir::LLVM::LLVMStructType>()) {
        // Descriptor logic
        llvm::SmallVector<int64_t, 1> ptrPos{0};
        mlir::Value ptr = builder_.create<mlir::LLVM::ExtractValueOp>(loc_, descriptor, ptrPos);
        
        mlir::Value linearIdx;
        auto i64Ty = builder_.getI64Type();
        
        if (indices.size() == 1) {
            linearIdx = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, indices[0]);
        } else if (indices.size() == 2) {
            mlir::Value i = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, indices[0]);
            mlir::Value j = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, indices[1]);
            
            llvm::SmallVector<int64_t, 1> colsPos{2};
            mlir::Value cols = builder_.create<mlir::LLVM::ExtractValueOp>(loc_, descriptor, colsPos);
            
            mlir::Value rowOffset = builder_.create<mlir::arith::MulIOp>(loc_, i, cols);
            linearIdx = builder_.create<mlir::arith::AddIOp>(loc_, rowOffset, j);
        } else {
             throw std::runtime_error("accessElement: unsupported index rank");
        }
        
        // For GEP, we need the element type.
        if (var->type.subTypes.empty()) {
             // Implicit string subtype (technically is empty)
             if (var->type.baseType == BaseType::STRING) {
                 // Implicit char type
             } else {
                 throw std::runtime_error("accessElement: variable has no subtype");
             }
        }
        
        mlir::Type elemTy;
        if (var->type.baseType == BaseType::STRING) {
            elemTy = builder_.getI8Type();
        } else {
            elemTy = getLLVMType(var->type.subTypes[0]);
        }
        
        mlir::Value gep = builder_.create<mlir::LLVM::GEPOp>(
            loc_, mlir::LLVM::LLVMPointerType::get(&context_),
            elemTy, ptr, mlir::ValueRange{linearIdx}
        );
        
        if (storeVal) {
            builder_.create<mlir::LLVM::StoreOp>(loc_, storeVal, gep);
            return nullptr;
        } else {
            return builder_.create<mlir::LLVM::LoadOp>(loc_, elemTy, gep);
        }
    }
    throw std::runtime_error("accessElement: variable is neither MemRef nor Descriptor");
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
    } else if (variable->type.baseType == BaseType::VECTOR || variable->type.baseType == BaseType::STRING) {
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

mlir::Value MLIRGen::computeArraySize(VarInfo* source, int line) {
    if (!source) {
        throw std::runtime_error("computeArraySize: source VarInfo is null");
    }
    
    // Check if source is an array literal (has compile-time known size)
    // For array literals, semantic analysis may have set dims[0] to the literal size
    // We create a runtime Value for it (even though it's compile-time known)
    if (source->type.baseType == BaseType::ARRAY && !source->type.dims.empty() && source->type.dims[0] >= 0) {
        // Static size from type - create constant index value
        return builder_.create<mlir::arith::ConstantIndexOp>(
            loc_, static_cast<int64_t>(source->type.dims[0]));
    }
    
    // For dynamic arrays or arrays where we need runtime size, use memref::DimOp
    // But first ensure the value exists
    if (!source->value) {
        throw std::runtime_error("computeArraySize: source value is null - array must be allocated first");
    }
    
    if (source->value.getType().isa<mlir::MemRefType>()) {
        auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        return builder_.create<mlir::memref::DimOp>(loc_, source->value, zeroIdx);
    }

    // Descriptor pointers (vector/string/array descriptors stored by pointer)
    if (auto ptrTy = source->value.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
        // Descriptors are stored via pointer to struct; element type is opaque on
        // opaque-pointer builds, so use the declared source type to recover the
        // struct layout.
        mlir::Type descTy = getLLVMType(source->type);
        if (auto structTy = descTy.dyn_cast<mlir::LLVM::LLVMStructType>()) {
            mlir::Value desc = builder_.create<mlir::LLVM::LoadOp>(loc_, structTy, source->value);
            llvm::SmallVector<int64_t, 1> lenPos{1};
            mlir::Value lenI64 = builder_.create<mlir::LLVM::ExtractValueOp>(loc_, desc, lenPos);
            return builder_.create<mlir::arith::IndexCastOp>(loc_, builder_.getIndexType(), lenI64);
        }
    }

    // Check if source is a slice struct (LLVM struct type)
    if (source->value.getType().isa<mlir::LLVM::LLVMStructType>()) {
        // Slice struct layout: { i32* ptr, i64 len }
        // Extract length from second field (index 1)
        // Note: This extraction happens in current block where slice struct exists
        // The returned size value can be used in entry block as long as current block
        // is reachable from entry block (which it should be for variable declarations)
        llvm::SmallVector<int64_t, 1> lenPos{1};
        mlir::Value sliceLen_i64 = builder_.create<mlir::LLVM::ExtractValueOp>(
            loc_, source->value, lenPos
        );
        // Convert i64 to index
        auto idxTy = builder_.getIndexType();
        return builder_.create<mlir::arith::IndexCastOp>(
            loc_, idxTy, sliceLen_i64
        );
    }
    std::string valTypeStr;
    {
        std::string tmp;
        llvm::raw_string_ostream rso(tmp);
        source->value.getType().print(rso);
        valTypeStr = rso.str();
    }
    throw std::runtime_error("computeArraySize: source is not a memref, slice struct, or array type. value type=" +
                             valTypeStr + ", source type=" + toString(source->type));
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

    // Ensure same dimension count (static check based on rank)
    if (lhs->runtimeDims.size() != rhs->runtimeDims.size()) {
        throw SizeError(line, "Mismatched lhs and rhs dimensions of " + std::to_string(lhs->runtimeDims.size()) + " and " + std::to_string(rhs->runtimeDims.size()));
    }

    // --- 2D Assignment (Matrix/Array) ---
    if (lhs->runtimeDims.size() == 2) {
        // Helper to get dimension size as Value (static or dynamic)
        auto getDim = [&](VarInfo* v, int idx) -> mlir::Value {
            if (v->type.dims.size() > (size_t)idx && v->type.dims[idx] >= 0) {
                return builder_.create<mlir::arith::ConstantIndexOp>(loc_, v->type.dims[idx]);
            }
            auto dVal = builder_.create<mlir::arith::ConstantIndexOp>(loc_, idx);
            return builder_.create<mlir::memref::DimOp>(loc_, v->value, dVal);
        };

        mlir::Value lRows = getDim(lhs, 0);
        mlir::Value lCols = getDim(lhs, 1);
        mlir::Value rRows = getDim(rhs, 0);
        mlir::Value rCols = getDim(rhs, 1);

        // Emit Runtime Check: lRows == rRows && lCols == rCols
        mlir::Value rowEq = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::eq, lRows, rRows);
        mlir::Value colEq = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::eq, lCols, rCols);
        mlir::Value dimEq = builder_.create<mlir::arith::AndIOp>(loc_, rowEq, colEq);

        auto ifOp = builder_.create<mlir::scf::IfOp>(loc_, dimEq, /*else*/ true);
        
        // Else block (Mismatch) -> Error
        builder_.setInsertionPointToStart(&ifOp.getElseRegion().front());
        {
             std::string errorMsgStr = "Runtime Error: Matrix assignment dimension mismatch";
             std::string errorMsgName = "assign_matrix_size_err_str";
             if (!module_.lookupSymbol<mlir::LLVM::GlobalOp>(errorMsgName)) {
                  mlir::OpBuilder moduleBuilder(module_.getBodyRegion());
                  mlir::Type charType = builder_.getI8Type();
                  auto strRef = mlir::StringRef(errorMsgStr.c_str(), errorMsgStr.length() + 1);
                  auto strType = mlir::LLVM::LLVMArrayType::get(charType, strRef.size());
                  moduleBuilder.create<mlir::LLVM::GlobalOp>(loc_, strType, true,
                                          mlir::LLVM::Linkage::Internal, errorMsgName,
                                          builder_.getStringAttr(strRef), 0);
             }
             auto globalErrorMsg = module_.lookupSymbol<mlir::LLVM::GlobalOp>(errorMsgName);
             mlir::Value errorMsgPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalErrorMsg);
             auto sizeErrorFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("SizeError");
             if (sizeErrorFunc) builder_.create<mlir::LLVM::CallOp>(loc_, sizeErrorFunc, mlir::ValueRange{errorMsgPtr});
             // Default yield already present at end of region.
        }

        // Then block (Match) -> Copy
        builder_.setInsertionPointToStart(&ifOp.getThenRegion().front());
        {
            auto c0 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
            auto c1 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

            builder_.create<mlir::scf::ForOp>(loc_, c0, lRows, c1, mlir::ValueRange{}, [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value i, mlir::ValueRange) {
                b.create<mlir::scf::ForOp>(l, c0, lCols, c1, mlir::ValueRange{}, [&](mlir::OpBuilder &b2, mlir::Location l2, mlir::Value j, mlir::ValueRange) {
                    mlir::Value val = b2.create<mlir::memref::LoadOp>(l2, rhs->value, mlir::ValueRange{i, j});
                    b2.create<mlir::memref::StoreOp>(l2, val, lhs->value, mlir::ValueRange{i, j});
                    b2.create<mlir::scf::YieldOp>(l2);
                });
                b.create<mlir::scf::YieldOp>(l);
            });
            // Default yield already present at end of region.
        }
        builder_.setInsertionPointAfter(ifOp);
        return;
    }

    // --- 1D Assignment ---
    // Handle 1D array/vector assignment
    if (lhs->runtimeDims.empty() || rhs->runtimeDims.empty()) {
        throw SizeError(line, "Missing runtime dimensions for array/vector assignment");
    }
    
    mlir::Value lhsLen, rhsLen;
    
    // Get LHS Length
    if (!lhs->type.dims.empty() && lhs->type.dims[0] >= 0) {
        lhsLen = builder_.create<mlir::arith::ConstantIndexOp>(loc_, lhs->type.dims[0]);
    } else {
        lhsLen = computeArraySize(lhs, line);
    }

    // Get RHS Length
    if (!rhs->type.dims.empty() && rhs->type.dims[0] >= 0) {
        rhsLen = builder_.create<mlir::arith::ConstantIndexOp>(loc_, rhs->type.dims[0]);
    } else {
        rhsLen = computeArraySize(rhs, line);
    }

    // Enforce same length assignment only, only exception being vectors (rhs is vector).
    // If rhs is vector, we allow mismatch (pad/trunc).
    if (rhs->type.baseType != BaseType::VECTOR) {
        // Strict check: lhsLen == rhsLen
        mlir::Value lenEq = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::eq, lhsLen, rhsLen);
        auto ifOp = builder_.create<mlir::scf::IfOp>(loc_, lenEq, /*else*/ true);
        
        builder_.setInsertionPointToStart(&ifOp.getElseRegion().front());
        {
             std::string errorMsgStr = "Runtime Error: Array assignment length mismatch";
             std::string errorMsgName = "assign_array_len_err_str";
             if (!module_.lookupSymbol<mlir::LLVM::GlobalOp>(errorMsgName)) {
                  mlir::OpBuilder moduleBuilder(module_.getBodyRegion());
                  mlir::Type charType = builder_.getI8Type();
                  auto strRef = mlir::StringRef(errorMsgStr.c_str(), errorMsgStr.length() + 1);
                  auto strType = mlir::LLVM::LLVMArrayType::get(charType, strRef.size());
                  moduleBuilder.create<mlir::LLVM::GlobalOp>(loc_, strType, true,
                                          mlir::LLVM::Linkage::Internal, errorMsgName,
                                          builder_.getStringAttr(strRef), 0);
             }
             auto globalErrorMsg = module_.lookupSymbol<mlir::LLVM::GlobalOp>(errorMsgName);
             mlir::Value errorMsgPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalErrorMsg);
             auto sizeErrorFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("SizeError");
             if (sizeErrorFunc) builder_.create<mlir::LLVM::CallOp>(loc_, sizeErrorFunc, mlir::ValueRange{errorMsgPtr});
             // Default yield already present at end of region.
        }
        // Then block just returns; default yield terminates region.
        builder_.setInsertionPointAfter(ifOp);
    }

    // Loop copy
    auto c0 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    auto c1 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

   builder_.create<mlir::scf::ForOp>(loc_, c0, lhsLen, c1, mlir::ValueRange{}, [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value idx, mlir::ValueRange) {
        // If RHS is vector, check bounds for padding/truncation
        if (rhs->type.baseType == BaseType::VECTOR) {
            auto inBounds = b.create<mlir::arith::CmpIOp>(l, mlir::arith::CmpIPredicate::slt, idx, rhsLen);
            auto ifBounds = b.create<mlir::scf::IfOp>(l, inBounds, /*else*/ true);
            
            // Clear default terminators to avoid duplicate yields
            mlir::Block &thenBlock = ifBounds.getThenRegion().front();
            mlir::Block &elseBlock = ifBounds.getElseRegion().front();
            thenBlock.clear();
            elseBlock.clear();

            // In bounds: load from RHS (descriptor-backed vector)
            {
                mlir::OpBuilder thenB(&thenBlock, thenBlock.end());
                mlir::Type elemTy = getLLVMType(rhs->type.subTypes[0]);
                auto descTy = getLLVMType(rhs->type);
                mlir::Value desc = thenB.create<mlir::LLVM::LoadOp>(l, descTy, rhs->value);
                llvm::SmallVector<int64_t, 1> ptrPos{0};
                mlir::Value ptr = thenB.create<mlir::LLVM::ExtractValueOp>(l, desc, ptrPos);
                auto i64Ty = thenB.getI64Type();
                mlir::Value idxI64 = thenB.create<mlir::arith::IndexCastOp>(l, i64Ty, idx);
                auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
                mlir::Value gep = thenB.create<mlir::LLVM::GEPOp>(
                    l, ptrTy, elemTy, ptr, mlir::ValueRange{idxI64});
                mlir::Value val = thenB.create<mlir::LLVM::LoadOp>(l, elemTy, gep);

                // Simple promotion check (int -> real)
                mlir::Value promotedVal = val;
                CompleteType srcT = rhs->type.subTypes[0];
                CompleteType dstT = lhs->type.subTypes[0];
                if (srcT.baseType == BaseType::INTEGER && dstT.baseType == BaseType::REAL) {
                    promotedVal = thenB.create<mlir::arith::SIToFPOp>(l, thenB.getF32Type(), val);
                }
                
                thenB.create<mlir::memref::StoreOp>(l, promotedVal, lhs->value, mlir::ValueRange{idx});
                thenB.create<mlir::scf::YieldOp>(l);
            }
            
            // Out of bounds: store zero
            {
                mlir::OpBuilder elseB(&elseBlock, elseBlock.end());
                mlir::Value zero;
                mlir::Type dstEleTy = getLLVMType(lhs->type.subTypes[0]);
                if (dstEleTy.isa<mlir::FloatType>())
                    zero = elseB.create<mlir::arith::ConstantOp>(l, dstEleTy, elseB.getFloatAttr(dstEleTy, 0.0));
                else
                    zero = elseB.create<mlir::arith::ConstantOp>(l, dstEleTy, elseB.getIntegerAttr(dstEleTy, 0));
                    
                elseB.create<mlir::memref::StoreOp>(l, zero, lhs->value, mlir::ValueRange{idx});
                elseB.create<mlir::scf::YieldOp>(l);
            }
            
            b.setInsertionPointAfter(ifBounds);
        } else {
            // Normal array copy - use accessElement and promoteType for consistency
            mlir::Value srcElemVal = accessElement(rhs, mlir::ValueRange{idx});
            
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
            accessElement(lhs, mlir::ValueRange{idx}, storeVal);
        }
        b.create<mlir::scf::YieldOp>(l);
    });
}

void MLIRGen::assignToVector(VarInfo* literal, VarInfo* variable, int line) {
    // Check types
    bool isString = (variable->type.baseType == BaseType::STRING);
    bool isVector = (variable->type.baseType == BaseType::VECTOR);
    
    if (!isString && !isVector) {
        throw TypeError(line, "assignToVector: target must be STRING or VECTOR");
    }
    
    // Check if variable is a mutable descriptor (pointer to LLVM struct)
    // Local variables allocated via allocaVar (for VECTOR/STRING) are pointers to descriptors.
    if (!variable->value.getType().isa<mlir::LLVM::LLVMPointerType>()) {
         throw std::runtime_error("assignToVector: variable must be a mutable descriptor (LLVM pointer)");
    }

    // determine length, handle both vector/struct rhs
    mlir::Value newLenVal;
    if (literal->type.baseType == BaseType::STRING) {        
        // If literal is descriptor (struct)
        if (literal->value.getType().isa<mlir::LLVM::LLVMStructType>()) {
             llvm::SmallVector<int64_t, 1> lenPos{1};
             newLenVal = builder_.create<mlir::LLVM::ExtractValueOp>(loc_, literal->value, lenPos);
             // Convert to index for alloc
             newLenVal = builder_.create<mlir::arith::IndexCastOp>(loc_, builder_.getIndexType(), newLenVal);
        } else if (literal->value.getType().isa<mlir::LLVM::LLVMPointerType>()) {
             // Pointer to descriptor
             mlir::Value desc = builder_.create<mlir::LLVM::LoadOp>(loc_, getLLVMType(literal->type), literal->value);
             llvm::SmallVector<int64_t, 1> lenPos{1};
             mlir::Value lenI64 = builder_.create<mlir::LLVM::ExtractValueOp>(loc_, desc, lenPos);
             newLenVal = builder_.create<mlir::arith::IndexCastOp>(loc_, builder_.getIndexType(), lenI64);
        } else {
             // Assume memref
             newLenVal = computeArraySize(literal, line);
        }
    } else {
        // Vector/Array
        // Resolve dims
        if (literal->runtimeDims.empty()) literal->runtimeDims = literal->type.dims;
        if (!literal->runtimeDims.empty() && literal->runtimeDims[0] >= 0) {
             newLenVal = builder_.create<mlir::arith::ConstantIndexOp>(loc_, literal->runtimeDims[0]);
        } else {
             newLenVal = computeArraySize(literal, line);
        }
    }
    
    // Alloca new memory
    mlir::Type elemTy;
    if (isString) elemTy = builder_.getI8Type();
    else elemTy = getLLVMType(variable->type.subTypes[0]);
    
    auto memTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
    mlir::Value newMemRef = builder_.create<mlir::memref::AllocaOp>(loc_, memTy, newLenVal);
    
    // Copy data to new memory
    auto c0 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    auto c1 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    
    builder_.create<mlir::scf::ForOp>(
        loc_, c0, newLenVal, c1, mlir::ValueRange{},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value iv, mlir::ValueRange args) {
            // Load from source
            mlir::Value elem = accessElement(literal, mlir::ValueRange{iv});
            
            // Store to new buffer
            b.create<mlir::memref::StoreOp>(l, elem, newMemRef, mlir::ValueRange{iv});
            
            b.create<mlir::scf::YieldOp>(l);
        }
    );
    
    // 4. Update descriptor
    // Extract ptr from newMemRef
    mlir::Value ptrAsIdx = builder_.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc_, newMemRef);
    auto i64Ty = builder_.getI64Type();
    mlir::Value ptrI64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, ptrAsIdx);
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
    mlir::Value ptr = builder_.create<mlir::LLVM::IntToPtrOp>(loc_, ptrTy, ptrI64);
    
    mlir::Value lenI64 = builder_.create<mlir::arith::IndexCastOp>(loc_, i64Ty, newLenVal);
    
    mlir::Type descTy = getLLVMType(variable->type);
    mlir::Value undef = builder_.create<mlir::LLVM::UndefOp>(loc_, descTy);
    mlir::Value desc = builder_.create<mlir::LLVM::InsertValueOp>(loc_, undef, ptr, llvm::SmallVector<int64_t, 1>{0});
    desc = builder_.create<mlir::LLVM::InsertValueOp>(loc_, desc, lenI64, llvm::SmallVector<int64_t, 1>{1});
    
    builder_.create<mlir::LLVM::StoreOp>(loc_, desc, variable->value);
    
    // Try to update runtime size, if can't then assume dynamic
    if (auto constLen = newLenVal.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        variable->runtimeDims = {static_cast<int>(constLen.value())};
    } else {
        variable->runtimeDims = {-1}; 
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

void MLIRGen::allocaVar(VarInfo* varInfo, int line, const std::vector<mlir::Value>& sizeValues) {
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
            if (varInfo->type.subTypes.empty()) {
                throw std::runtime_error("allocaVar: ARRAY type must have at least one element subtype");
            }
            
            mlir::Type elemTy = getLLVMType(varInfo->type.subTypes[0]);
            
            if (!elemTy) {
                throw std::runtime_error("allocaVar: failed to get LLVM type for array element type");
            }
            
            if (varInfo->type.dims.size() == 1 && varInfo->type.dims[0] >= 0) {
                // Static array size
                int64_t n = varInfo->type.dims[0];
                auto memTy = mlir::MemRefType::get({n}, elemTy);
                varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                    loc_, memTy, mlir::ValueRange{}, mlir::ValueRange{}, builder_.getIntegerAttr(builder_.getI64Type(), 16));
            } else if (varInfo->type.dims.size() == 1 && varInfo->type.dims[0] < 0) {
                // Dynamic array size - requires sizeValue parameter
                if (sizeValues.size() != 1) {
                    throw std::runtime_error("allocaVar: dynamic array requires sizeValue parameter for variable '" +
                                             (varInfo->identifier.empty() ? std::string("<temporary>") : varInfo->identifier) + "'");
                }
                // For dynamic arrays, allocate in current block (not entry block) because
                // sizeValue is computed in current block and must dominate the allocation
                // This is safe for variable declarations which happen early in function body
                auto memTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
                varInfo->value = builder_.create<mlir::memref::AllocaOp>(
                    loc_,
                    memTy,
                    mlir::ValueRange{sizeValues[0]}, // dynamic size operand
                    mlir::ValueRange{},           // symbol operands
                    builder_.getIntegerAttr(builder_.getI64Type(), 16)           // alignment
                );
            } else if (varInfo->type.dims.size() == 2 && varInfo->type.dims[0] >= 0 && varInfo->type.dims[1] >= 0) {
                // 2D array (matrix)
                int64_t rows = varInfo->type.dims[0];
                int64_t cols = varInfo->type.dims[1];
                auto memTy = mlir::MemRefType::get({rows, cols}, elemTy);
                varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                    loc_, memTy, mlir::ValueRange{}, mlir::ValueRange{}, builder_.getIntegerAttr(builder_.getI64Type(), 16));
            } else if (varInfo->type.dims.size() == 2 &&
                       (varInfo->type.dims[0] < 0 || varInfo->type.dims[1] < 0)) {
                // Dynamic 2D
                if (sizeValues.size() != 2) {
                    throw std::runtime_error("allocaVar: dynamic 2D array requires two size operands for variable '" +
                                             (varInfo->identifier.empty() ? std::string("<temporary>") : varInfo->identifier) + "'");
                }
                auto memTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, elemTy);
                llvm::ArrayRef<mlir::Value> dynSizes(sizeValues);
                varInfo->value = builder_.create<mlir::memref::AllocaOp>(
                    loc_,
                    memTy,
                    dynSizes, // two dynamic sizes
                    mlir::ValueRange{},
                    builder_.getIntegerAttr(builder_.getI64Type(), 16));
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
                if (sizeValues.size() != 2) {
                    throw std::runtime_error("allocaVar: MATRIX dynamic dimensions require two size operands for variable '" +
                                             (varInfo->identifier.empty() ? std::string("<unknown>") : varInfo->identifier) + "'");
                }
                auto memTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, elemTy);
                llvm::ArrayRef<mlir::Value> dynSizes(sizeValues);
                varInfo->value = builder_.create<mlir::memref::AllocaOp>(
                    loc_, memTy, dynSizes, mlir::ValueRange{}, mlir::IntegerAttr());
            } else {
                int64_t rows = varInfo->type.dims[0];
                int64_t cols = varInfo->type.dims[1];
                auto memTy = mlir::MemRefType::get({rows, cols}, elemTy);
                varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(loc_, memTy);
            }
            break;
        }

        case BaseType::VECTOR:
        case BaseType::STRING: {
            mlir::Type descTy = getLLVMType(varInfo->type);
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
            auto i64Ty = builder_.getI64Type();
            auto oneAttr = builder_.getIntegerAttr(i64Ty, 1);
            mlir::Value one = entryBuilder.create<mlir::arith::ConstantOp>(loc_, i64Ty, oneAttr);
            varInfo->value = entryBuilder.create<mlir::LLVM::AllocaOp>(loc_, ptrTy, descTy, one, 0u);
            
            // Initialize with {null, 0}
            mlir::Value undef = entryBuilder.create<mlir::LLVM::UndefOp>(loc_, descTy);
            mlir::Value nullPtr = entryBuilder.create<mlir::LLVM::ZeroOp>(loc_, ptrTy);
            mlir::Value zero = entryBuilder.create<mlir::arith::ConstantOp>(loc_, i64Ty, builder_.getIntegerAttr(i64Ty, 0));
            
            llvm::SmallVector<int64_t, 1> ptrPos{0};
            mlir::Value desc = entryBuilder.create<mlir::LLVM::InsertValueOp>(loc_, undef, nullPtr, ptrPos);
            llvm::SmallVector<int64_t, 1> lenPos{1};
            desc = entryBuilder.create<mlir::LLVM::InsertValueOp>(loc_, desc, zero, lenPos);
            
            entryBuilder.create<mlir::LLVM::StoreOp>(loc_, desc, varInfo->value);
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
    // Legacy overload: convert int to Value and delegate
    // Note: use entry block builder? The old implementation used entryBuilder.
    // The new Value-based implementation uses builder_ (current point).
    // To maintain old behavior of static allocs being at top, we check:
    
    mlir::func::FuncOp funcOp = getCurrentEnclosingFunction();
    mlir::Block &entry = funcOp.front();
    mlir::OpBuilder entryBuilder(&entry, entry.begin());
    
    mlir::Value arrayLen = entryBuilder.create<mlir::arith::ConstantOp>(
        loc_, builder_.getIndexType(), builder_.getIntegerAttr(builder_.getIndexType(), len));
    
    return allocaVector(arrayLen, varInfo);
}

mlir::Value MLIRGen::allocaVector(mlir::Value len, VarInfo *varInfo) {
    // For dynamic length, we must use current builder
    // For static length (passed as ConstantOp Value), we *could* use entry builder, but 
    // since we receive a Value here, we assume it's available at current insertion point.
    // We create the alloca at the current insertion point to support dynamic sizing.
    // Warning: Don't call this in a loop unless you mean to explode the stack.
    
    mlir::Type elemTy = getLLVMType(varInfo->type.subTypes[0]);
    auto memTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
    
    auto alloca = builder_.create<mlir::memref::AllocaOp>(
        loc_,
        memTy,
        mlir::ValueRange{len},
        mlir::ValueRange{},
        builder_.getIntegerAttr(builder_.getI64Type(), 16));
    
    varInfo->value = alloca;
    return alloca;
}

void MLIRGen::zeroInitializeVar(VarInfo* var) {
    if (!var->value) return;

    if (auto memrefTy = var->value.getType().dyn_cast<mlir::MemRefType>()) {
        if (memrefTy.getRank() > 0) {
            // Avoid emitting rank-mismatched store; skip zero-fill for shaped memrefs here.
            return;
        }
    }

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