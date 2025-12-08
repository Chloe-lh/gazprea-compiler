#include "CompileTimeExceptions.h"
#include "MLIRgen.h"

void MLIRGen::storeZeroElement(mlir::Value destMemRef,
                               const CompleteType &elemType,
                               mlir::ValueRange indices) {
    BaseType elemBase = elemType.baseType;
    mlir::Type elemTy = getLLVMType(elemType);
    mlir::Value zero;

    if (elemBase == BaseType::INTEGER ||
        elemBase == BaseType::BOOL ||
        elemBase == BaseType::CHARACTER) {
        zero = builder_.create<mlir::arith::ConstantOp>(
            loc_, elemTy, builder_.getIntegerAttr(elemTy, 0));
    } else if (elemBase == BaseType::REAL) {
        zero = builder_.create<mlir::arith::ConstantOp>(
            loc_, elemTy, builder_.getFloatAttr(elemTy, 0.0));
    } else {
        throw std::runtime_error("MLIRGen::storeZeroElement: unsupported element type for zero padding");
    }

    builder_.create<mlir::memref::StoreOp>(loc_, zero, destMemRef, indices);
}

mlir::Value MLIRGen::loadElementByFlatIndex(VarInfo *from, int64_t flatIndex, int srcRank, int line) {
    if (flatIndex < 0) {
        return mlir::Value();
    }

    auto idxTy = builder_.getIndexType();

    if (srcRank == 1) {
        if (from->runtimeDims.empty()) {
            from->runtimeDims = from->type.dims;
        }
        int64_t srcLen = from->runtimeDims.empty() ? -1 : from->runtimeDims[0];
        if (srcLen < 0 || flatIndex >= srcLen) {
            return mlir::Value();
        }

        mlir::Value idx = builder_.create<mlir::arith::ConstantOp>(
            loc_, idxTy, builder_.getIntegerAttr(idxTy, flatIndex));
        return builder_.create<mlir::memref::LoadOp>(
            loc_, from->value, mlir::ValueRange{idx});
    }

    if (srcRank == 2) {
        if (from->runtimeDims.size() < 2) {
            from->runtimeDims = from->type.dims;
        }
        if (from->runtimeDims.size() < 2) {
            throw SizeError(line, "MLIRGen::loadElementByFlatIndex: invalid source matrix dimensions");
        }
        int64_t srcRows = from->runtimeDims[0];
        int64_t srcCols = from->runtimeDims[1];
        if (srcRows < 0 || srcCols < 0) {
            throw SizeError(line, "MLIRGen::loadElementByFlatIndex: invalid source matrix dimensions");
        }
        int64_t total = srcRows * srcCols;
        if (flatIndex >= total) {
            return mlir::Value();
        }

        int64_t row = flatIndex / srcCols;
        int64_t col = flatIndex % srcCols;

        mlir::Value rIdx = builder_.create<mlir::arith::ConstantOp>(
            loc_, idxTy, builder_.getIntegerAttr(idxTy, row));
        mlir::Value cIdx = builder_.create<mlir::arith::ConstantOp>(
            loc_, idxTy, builder_.getIntegerAttr(idxTy, col));
        return builder_.create<mlir::memref::LoadOp>(
            loc_, from->value, mlir::ValueRange{rIdx, cIdx});
    }

    // Unsupported rank
    return mlir::Value();
}

void MLIRGen::expandScalarToAggregate(VarInfo *from,
                                      VarInfo &to,
                                      CompleteType *toType,
                                      int line) {
    if (!isScalarType(from->type.baseType)) {
        throw std::runtime_error("MLIRGen::expandScalarToAggregate: source must be scalar");
    }

    if (toType->subTypes.size() != 1) {
        throw std::runtime_error("MLIRGen::expandScalarToAggregate: aggregate must have exactly one element subtype");
    }

    const CompleteType &elemType = toType->subTypes[0];

    if (!isScalarType(elemType.baseType) ||
        !canScalarCast(from->type.baseType, elemType.baseType)) {
        throw TypeError(line, std::string("Codegen: cannot cast from '") +
                               toString(from->type) + "' to '" +
                               toString(*toType) + "'.");
    }

    auto idxTy = builder_.getIndexType();

    if (toType->baseType == BaseType::VECTOR) {
        // Broadcast scalar to all vector elements
        int len = 1;
        if (!toType->dims.empty() && toType->dims[0] > 0) {
            len = toType->dims[0];
        }
        to.runtimeDims = {len};

        mlir::Value newVec = allocaVector(len, &to);
        to.value = newVec;

        VarInfo elemVar = castType(from, const_cast<CompleteType*>(&elemType), line);
        mlir::Value elemVal = getSSAValue(elemVar);

        for (int i = 0; i < len; ++i) {
            mlir::Value idx = builder_.create<mlir::arith::ConstantOp>(
                loc_, idxTy,
                builder_.getIntegerAttr(idxTy, static_cast<int64_t>(i)));
            builder_.create<mlir::memref::StoreOp>(
                loc_, elemVal, to.value, mlir::ValueRange{idx});
        }
        return;
    }

    // Arrays and matrices: use static dimensions from type.
    if (toType->dims.empty()) {
        throw SizeError(line, "MLIRGen::expandScalarToAggregate: destination aggregate has no dimensions");
    }

    if (toType->dims.size() == 1) {
        int64_t len = toType->dims[0];
        if (len < 0) {
            throw SizeError(line, "MLIRGen::expandScalarToAggregate: invalid 1D length");
        }

        for (int64_t i = 0; i < len; ++i) {
            mlir::Value idx = builder_.create<mlir::arith::ConstantOp>(
                loc_, idxTy, builder_.getIntegerAttr(idxTy, i));

            VarInfo elemVar = castType(from, const_cast<CompleteType*>(&elemType), line);
            mlir::Value elemVal = getSSAValue(elemVar);

            builder_.create<mlir::memref::StoreOp>(
                loc_, elemVal, to.value, mlir::ValueRange{idx});
        }
    } else if (toType->dims.size() == 2) {
        int64_t rows = toType->dims[0];
        int64_t cols = toType->dims[1];
        if (rows < 0 || cols < 0) {
            throw SizeError(line, "MLIRGen::expandScalarToAggregate: invalid 2D dimensions");
        }

        for (int64_t i = 0; i < rows; ++i) {
            mlir::Value rowIdx = builder_.create<mlir::arith::ConstantOp>(
                loc_, idxTy, builder_.getIntegerAttr(idxTy, i));
            for (int64_t j = 0; j < cols; ++j) {
                mlir::Value colIdx = builder_.create<mlir::arith::ConstantOp>(
                    loc_, idxTy, builder_.getIntegerAttr(idxTy, j));

                VarInfo elemVar = castType(from, const_cast<CompleteType*>(&elemType), line);
                mlir::Value elemVal = getSSAValue(elemVar);

                builder_.create<mlir::memref::StoreOp>(
                    loc_, elemVal, to.value, mlir::ValueRange{rowIdx, colIdx});
            }
        }
    } else {
        throw SizeError(line, "MLIRGen::expandScalarToAggregate: unsupported aggregate rank");
    }
}

//* Only allows implicit promotion from integer -> real. throws AssignError otherwise. */
VarInfo MLIRGen::promoteType(VarInfo* from, CompleteType* toType, int line) {
    if (toType->baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("promoteType: target type is UNKNOWN");
    }
    if (from->type.baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("promoteType: source type is UNKNOWN");
    }
    // No-op when types are identical
    if (from->type == *toType) {
        return *from;
    }

    // TODO: support scalar -> array/matrix promotion

    // Only support integer -> real promotion
    if (from->type.baseType == BaseType::INTEGER && toType->baseType == BaseType::REAL) {
        VarInfo to = VarInfo(*toType);
        allocaLiteral(&to, line);
        mlir::Value i32Val = getSSAValue(*from);
        mlir::Value fVal = builder_.create<mlir::arith::SIToFPOp>(loc_, builder_.getF32Type(), i32Val);
        builder_.create<mlir::memref::StoreOp>(loc_, fVal, to.value, mlir::ValueRange{});
        return to;
    }

    throw AssignError(line, std::string("Codegen: unsupported promotion from '") +
        toString(from->type) + "' to '" + toString(*toType) + "'.");
}

VarInfo MLIRGen::castType(VarInfo* from, CompleteType* toType, int line) {
    VarInfo to = VarInfo(*toType);
    allocaLiteral(&to, line); // Create new value container

    switch (from->type.baseType) {
        case (BaseType::STRUCT):
        {
            // Reject casting struct to non-struct types
            if (toType->baseType != BaseType::STRUCT) {
                throw TypeError(line, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            // Ensure same len members
            if (from->type.subTypes.size() != toType->subTypes.size()) {
                throw TypeError(
                    line, std::string("Codegen: cannot cast mismatched sizes from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }

            if (!from->value) {
                allocaVar(from, line);
            }

            // Load source struct and initialise destination struct as undef
            mlir::Type srcStructTy = getLLVMType(from->type);
            mlir::Value srcStruct = builder_.create<mlir::LLVM::LoadOp>(
                loc_, srcStructTy, from->value);
            mlir::Type dstStructTy = getLLVMType(*toType);
            mlir::Value dstStruct =
                builder_.create<mlir::LLVM::UndefOp>(loc_, dstStructTy);

            try {
                for (size_t i = 0; i < from->type.subTypes.size(); ++i) {
                    llvm::SmallVector<int64_t, 1> pos{
                        static_cast<int64_t>(i)};
                    // Extract element from source tuple
                    mlir::Value srcElem =
                        builder_.create<mlir::LLVM::ExtractValueOp>(
                            loc_, srcStruct, pos);

                    // Wrap element in a VarInfo so we can reuse scalar casting
                    VarInfo fromElem(from->type.subTypes[i]);
                    allocaLiteral(&fromElem, line);
                    builder_.create<mlir::memref::StoreOp>(
                        loc_, srcElem, fromElem.value, mlir::ValueRange{});

                    VarInfo castedElem =
                        castType(&fromElem, &toType->subTypes[i], line);
                    mlir::Value elemVal = getSSAValue(castedElem);

                    // Insert casted element into destination struct
                    dstStruct = builder_.create<mlir::LLVM::InsertValueOp>(
                        loc_, dstStruct, elemVal, pos);
                }
            } catch (TypeError &le) {
                throw TypeError(
                    line, std::string("Codegen: cannot cast from '") +
                           toString(from->type) + "' to '" +
                           toString(*toType) + "':\n\n" + le.what());
            }

            // Store constructed struct into destination tuple storage
            builder_.create<mlir::LLVM::StoreOp>(
                loc_, dstStruct, to.value);
            break;
        }
        case (BaseType::TUPLE):
        {
            // Reject casting tuple to non-tuple types
            if (toType->baseType != BaseType::TUPLE) {
                throw TypeError(line, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            // Ensure same length tuples
            if (from->type.subTypes.size() != toType->subTypes.size()) {
                throw TypeError(
                    line, std::string("Codegen: cannot cast mismatched sizes from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }

            if (!from->value) {
                allocaVar(from, line);
            }

            // Load source struct and initialise destination struct as undef
            mlir::Type srcStructTy = getLLVMType(from->type);
            mlir::Value srcStruct = builder_.create<mlir::LLVM::LoadOp>(
                loc_, srcStructTy, from->value);
            mlir::Type dstStructTy = getLLVMType(*toType);
            mlir::Value dstStruct =
                builder_.create<mlir::LLVM::UndefOp>(loc_, dstStructTy);

            try {
                for (size_t i = 0; i < from->type.subTypes.size(); ++i) {
                    llvm::SmallVector<int64_t, 1> pos{
                        static_cast<int64_t>(i)};
                    // Extract element from source tuple
                    mlir::Value srcElem =
                        builder_.create<mlir::LLVM::ExtractValueOp>(
                            loc_, srcStruct, pos);

                    // Wrap element in a VarInfo so we can reuse scalar casting
                    VarInfo fromElem(from->type.subTypes[i]);
                    allocaLiteral(&fromElem, line);
                    builder_.create<mlir::memref::StoreOp>(
                        loc_, srcElem, fromElem.value, mlir::ValueRange{});

                    VarInfo castedElem =
                        castType(&fromElem, &toType->subTypes[i], line);
                    mlir::Value elemVal = getSSAValue(castedElem);

                    // Insert casted element into destination struct
                    dstStruct = builder_.create<mlir::LLVM::InsertValueOp>(
                        loc_, dstStruct, elemVal, pos);
                }
            } catch (TypeError &le) {
                throw TypeError(
                    line, std::string("Codegen: cannot cast from '") +
                           toString(from->type) + "' to '" +
                           toString(*toType) + "':\n\n" + le.what());
            }

            // Store constructed struct into destination tuple storage
            builder_.create<mlir::LLVM::StoreOp>(
                loc_, dstStruct, to.value);
            break;
        }
        case (BaseType::BOOL):
        {
            mlir::Value boolVal = getSSAValue(*from); // Load value or use SSA
            switch (toType->baseType) {
                case BaseType::ARRAY:
                case BaseType::VECTOR:
                case BaseType::MATRIX:
                    expandScalarToAggregate(from, to, toType, line);
                    break;

                case BaseType::BOOL:                    // Bool -> Bool
                    builder_.create<mlir::memref::StoreOp>(
                        loc_, boolVal, to.value, mlir::ValueRange{});
                    break;
       
                case BaseType::INTEGER:                 // Bool -> Int
                {
                    mlir::Value intVal = builder_.create<mlir::arith::ExtUIOp>(
                            loc_, builder_.getI32Type(), boolVal
                        );
                    builder_.create<mlir::memref::StoreOp>(loc_, intVal, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::CHARACTER:               // Bool -> Char
                {
                    mlir::Value charVal = builder_.create<mlir::arith::ExtUIOp>(
                        loc_, builder_.getI8Type(), boolVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, charVal, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::REAL:                    // Bool -> Real
                {
                    mlir::Value realVal = builder_.create<mlir::arith::UIToFPOp>(
                        loc_, builder_.getF32Type(), boolVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, realVal, to.value, mlir::ValueRange{});
                    break;
                }

                default:
                    throw TypeError(line, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            break;
        }

        case (BaseType::CHARACTER):
        {
            mlir::Value chVal = getSSAValue(*from);
            switch (toType->baseType) {
                case BaseType::ARRAY:
                case BaseType::VECTOR:
                case BaseType::MATRIX:
                    expandScalarToAggregate(from, to, toType, line);
                    break;

                case BaseType::CHARACTER:               // Char -> Char
                    builder_.create<mlir::memref::StoreOp>(loc_, chVal, to.value, mlir::ValueRange{});
                    break;

                case BaseType::BOOL:                    // Char -> Bool
                {
                    mlir::Value zeroConst = builder_.create<mlir::arith::ConstantOp>(
                        loc_, builder_.getI8Type(), builder_.getIntegerAttr(builder_.getI8Type(), 0)
                    );
                    mlir::Value isZeroConst = builder_.create<mlir::arith::CmpIOp>(
                        loc_, mlir::arith::CmpIPredicate::ne, chVal, zeroConst
                    );  // '\0' == false
                    builder_.create<mlir::memref::StoreOp>(loc_, isZeroConst, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::INTEGER:                 // Char -> Int
                {
                    mlir::Value intVal = builder_.create<mlir::arith::ExtUIOp>(
                            loc_, builder_.getI32Type(), chVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, intVal, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::REAL:                    // Char -> Real
                {
                    mlir::Value realVal = builder_.create<mlir::arith::UIToFPOp>(
                            loc_, builder_.getF32Type(), chVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, realVal, to.value, mlir::ValueRange{});
                    break;
                }

                default:
                    throw TypeError(line, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            break;
        }

        case (BaseType::INTEGER):
        {
            mlir::Value i32Val = getSSAValue(*from);
            switch (toType->baseType) {
                case BaseType::ARRAY:
                case BaseType::VECTOR:
                case BaseType::MATRIX:
                    expandScalarToAggregate(from, to, toType, line);
                    break;

                case BaseType::INTEGER:                 // Int -> Int
                    builder_.create<mlir::memref::StoreOp>(loc_, i32Val, to.value, mlir::ValueRange{});
                    break;

                case BaseType::BOOL:                    // Int -> Bool (ne 0)
                {
                    mlir::Value zero = builder_.create<mlir::arith::ConstantOp>(
                        loc_, builder_.getI32Type(), builder_.getIntegerAttr(builder_.getI32Type(), 0)
                    );
                    mlir::Value neZeroConstant = builder_.create<mlir::arith::CmpIOp>(
                        loc_, mlir::arith::CmpIPredicate::ne, i32Val, zero
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, neZeroConstant, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::CHARACTER:               // Int -> Char (mod 256)
                {
                    mlir::Value i8Val = builder_.create<mlir::arith::TruncIOp>(
                        loc_, builder_.getI8Type(), i32Val
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, i8Val, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::REAL:                    // Int -> Real
                {
                    mlir::Value fVal = builder_.create<mlir::arith::SIToFPOp>(
                        loc_, builder_.getF32Type(), i32Val
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, fVal, to.value, mlir::ValueRange{});
                    break;
                }

                default:
                    throw TypeError(line, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            break;
        }

        case (BaseType::REAL):
        {
            mlir::Value fVal = getSSAValue(*from);
            switch (toType->baseType) {
                case BaseType::ARRAY:
                case BaseType::VECTOR:
                case BaseType::MATRIX:
                    expandScalarToAggregate(from, to, toType, line);
                    break;

                case BaseType::REAL:                    // Real -> Real
                    builder_.create<mlir::memref::StoreOp>(loc_, fVal, to.value, mlir::ValueRange{});
                    break;

                case BaseType::INTEGER:                 // Real -> Int (truncate)
                {
                    mlir::Value iVal = builder_.create<mlir::arith::FPToSIOp>(
                        loc_, builder_.getI32Type(), fVal
                    );
                    builder_.create<mlir::memref::StoreOp>(loc_, iVal, to.value, mlir::ValueRange{});
                    break;
                }

                case BaseType::CHARACTER:               // Real -> Char (not allowed)
                case BaseType::BOOL:                    // Real -> Bool (not allowed)
                default:
                    throw TypeError(line, std::string("Codegen: cannot cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
            }
            break;
        }
        case (BaseType::ARRAY):
        case (BaseType::VECTOR):
        case (BaseType::MATRIX):
        {
            if (toType->baseType != BaseType::ARRAY &&
                toType->baseType != BaseType::VECTOR &&
                toType->baseType != BaseType::MATRIX) {
                throw TypeError(line, "MLIRGen::castType: cannot cast from '" + toString(from->type) + "' to '" + toString(*toType) + "'.");
                // TODO: confirm if downcasting from 1D/2D types to scalar is allowed
            }

            if (!from->value) {
                allocaVar(from, line);
            }

            if (from->type.subTypes.size() != 1 || toType->subTypes.size() != 1) {
                throw std::runtime_error("MLIRGen::castType: array/vector types must have exactly one element subtype");
            }

            // Ensure runtime dimensions exist
            if (from->runtimeDims.empty()) {
                from->runtimeDims = from->type.dims;
            }
            if (to.runtimeDims.empty()) {
                to.runtimeDims = toType->dims;
            }

            const CompleteType &fromElemCT = from->type.subTypes[0];
            const CompleteType &toElemCT   = toType->subTypes[0];

            int srcRank = static_cast<int>(from->runtimeDims.size());
            int dstRank = static_cast<int>(toType->dims.size());

            if (srcRank <= 0 || srcRank > 2 || dstRank <= 0 || dstRank > 2) {
                throw SizeError(line, "MLIRGen::castType: unsupported array/vector/matrix rank for cast");
            }
            auto idxTy = builder_.getIndexType();

            
            if (dstRank == 1) {                                 // Case 1: Casting to 1D type
                int64_t destLen = 0;

                // Calculate 
                if (toType->baseType == BaseType::VECTOR) {
                    // For vectors, take the total number of elements from the source
                    int64_t srcTotal = 0;
                    if (srcRank == 1) {
                        srcTotal = from->runtimeDims[0];
                    } else { // srcRank == 2
                        int64_t srcRows = from->runtimeDims[0];
                        int64_t srcCols = from->runtimeDims[1];

                        if (srcRows < 0 || srcCols < 0) throw SizeError(line, "MLIRGen::castType: invalid source matrix dimensions for vector cast");

                        srcTotal = srcRows * srcCols;
                    }
                    if (srcTotal < 0) {
                        throw SizeError(line, "MLIRGen::castType: negative source length for vector cast");
                    }

                    destLen = srcTotal;
                    to.runtimeDims = {static_cast<int>(destLen)};

                    // Re-allocate vector storage to the desired length
                    mlir::Value newVec = allocaVector(static_cast<int>(destLen), &to);
                    to.value = newVec;
                } else {
                    // 1D array: use declared dimension
                    if (toType->dims.empty() || toType->dims[0] < 0) {
                        throw SizeError(line, "MLIRGen::castType: invalid destination array length for cast");
                    }
                    destLen = toType->dims[0];
                    to.runtimeDims = {toType->dims[0]};
                }

                for (int64_t t = 0; t < destLen; ++t) {
                    // Determine source element using row-major indexing
                    mlir::Value srcVal = loadElementByFlatIndex(from, t, srcRank, line);

                    mlir::Value destIdx =
                        builder_.create<mlir::arith::ConstantOp>(
                            loc_, idxTy,
                            builder_.getIntegerAttr(idxTy, t));

                    if (srcVal) {
                        // Wrap source element to reuse castType recursively
                        VarInfo srcElemVar(fromElemCT);
                        allocaLiteral(&srcElemVar, line);
                        builder_.create<mlir::memref::StoreOp>(
                            loc_, srcVal, srcElemVar.value, mlir::ValueRange{});

                        VarInfo castedElem =
                            castType(&srcElemVar, const_cast<CompleteType*>(&toElemCT), line);
                        mlir::Value elemVal = getSSAValue(castedElem);

                        builder_.create<mlir::memref::StoreOp>(
                            loc_, elemVal, to.value, mlir::ValueRange{destIdx});
                    } else {
                        storeZeroElement(to.value, toElemCT, mlir::ValueRange{destIdx});
                    }
                }
            } else {                                        // 2. Handle matrix destination
                if (toType->dims.size() < 2 ||
                    toType->dims[0] < 0 || toType->dims[1] < 0) {
                    throw SizeError(line, "MLIRGen::castType: invalid destination matrix dimensions for cast");
                }

                int64_t dstRows = toType->dims[0];
                int64_t dstCols = toType->dims[1];
                to.runtimeDims = {toType->dims[0], toType->dims[1]};

                int64_t srcRows = 0;
                int64_t srcCols = 0;
                if (srcRank == 2) {
                    srcRows = from->runtimeDims[0];
                    srcCols = from->runtimeDims[1];
                    if (srcRows < 0 || srcCols < 0) {
                        throw SizeError(line, "MLIRGen::castType: invalid source matrix dimensions");
                    }
                } else { // srcRank == 1
                    if (from->runtimeDims[0] < 0) {
                        throw SizeError(line, "MLIRGen::castType: invalid source vector length for matrix cast");
                    }
                }

                for (int64_t i = 0; i < dstRows; ++i) {
                    mlir::Value rowIdx =
                        builder_.create<mlir::arith::ConstantOp>(
                            loc_, idxTy,
                            builder_.getIntegerAttr(idxTy, i));
                    for (int64_t j = 0; j < dstCols; ++j) {
                        mlir::Value colIdx =
                            builder_.create<mlir::arith::ConstantOp>(
                                loc_, idxTy,
                                builder_.getIntegerAttr(idxTy, j));

                        bool hasSrc = false;
                        mlir::Value srcVal;

                        if (srcRank == 2) {
                            if (i < srcRows && j < srcCols) {
                                hasSrc = true;
                                srcVal = builder_.create<mlir::memref::LoadOp>(
                                    loc_, from->value,
                                    mlir::ValueRange{rowIdx, colIdx});
                            }
                        } else { // srcRank == 1 => flatten into matrix row-major
                            int64_t srcLen = from->runtimeDims[0];
                            int64_t flatIndex = i * dstCols + j;
                            if (flatIndex < srcLen) {
                                hasSrc = true;
                                mlir::Value flatIdx =
                                    builder_.create<mlir::arith::ConstantOp>(
                                        loc_, idxTy,
                                        builder_.getIntegerAttr(idxTy, flatIndex));
                                srcVal = builder_.create<mlir::memref::LoadOp>(
                                    loc_, from->value,
                                    mlir::ValueRange{flatIdx});
                            }
                        }

                        if (hasSrc) {
                            VarInfo srcElemVar(fromElemCT);
                            allocaLiteral(&srcElemVar, line);
                            builder_.create<mlir::memref::StoreOp>(
                                loc_, srcVal, srcElemVar.value, mlir::ValueRange{});

                            VarInfo castedElem =
                                castType(&srcElemVar, const_cast<CompleteType*>(&toElemCT), line);
                            mlir::Value elemVal = getSSAValue(castedElem);

                            builder_.create<mlir::memref::StoreOp>(
                                loc_, elemVal, to.value,
                                mlir::ValueRange{rowIdx, colIdx});
                        } else {
                            storeZeroElement(to.value, toElemCT, mlir::ValueRange{rowIdx, colIdx});
                        }
                    }
                }
            }

            break;
        }

        default:
            throw TypeError(line, std::string("Codegen: unsupported cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
    }

    return to;
}
