#include "CompileTimeExceptions.h"
#include "MLIRgen.h"

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

        default:
            throw TypeError(line, std::string("Codegen: unsupported cast from '") + toString(from->type) + "' to '" + toString(*toType) + "'.");
    }

    return to;
}
