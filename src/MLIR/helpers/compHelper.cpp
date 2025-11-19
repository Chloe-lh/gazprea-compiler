#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

// Helper functions for comparison operations and equality
#include "MLIR/helpers/compHelper.h"
#include <stdexcept>
#include <llvm/ADT/SmallVector.h>

mlir::Value mlirScalarEquals(mlir::Value left, mlir::Value right, mlir::Location loc, mlir::OpBuilder& builder) {
    mlir::Type type = left.getType();
    if (type.isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, left, right);
    } else if (type.isa<mlir::FloatType>()) {
        return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, left, right);
    } else {
        throw std::runtime_error("FATAL: mlirScalarEquals: Unsupported type for equality");
    }
}

// Equality for LLVM struct-based tuples (and nested aggregates)
mlir::Value mlirAggregateEquals(mlir::Value left, mlir::Value right,
                                mlir::Location loc, mlir::OpBuilder &builder) {
    auto structTy = left.getType().dyn_cast<mlir::LLVM::LLVMStructType>();
    if (!structTy) {
        // Fallback to scalar equality for non-aggregate values
        return mlirScalarEquals(left, right, loc, builder);
    }

    auto body = structTy.getBody();
    if (body.empty()) {
        throw std::runtime_error(
            "FATAL: mlirAggregateEquals: empty struct type is invalid");
    }

    mlir::Value result;
    for (unsigned i = 0; i < body.size(); ++i) {
        llvm::SmallVector<int64_t, 1> pos{static_cast<int64_t>(i)};
        mlir::Value leftElem =
            builder.create<mlir::LLVM::ExtractValueOp>(loc, left, pos);
        mlir::Value rightElem =
            builder.create<mlir::LLVM::ExtractValueOp>(loc, right, pos);

        mlir::Value elemEq =
            mlirAggregateEquals(leftElem, rightElem, loc, builder);

        if (!result) {
            result = elemEq;
        } else {
            result =
                builder.create<mlir::arith::AndIOp>(loc, result, elemEq);
        }
    }
    return result;
}

