// Helper declarations for MLIR compare utilities
#pragma once

#include "mlir/IR/Builders.h"
#include <mlir/IR/Value.h>
#include <mlir/IR/Location.h>


// Compare two scalar MLIR values (integers/floats)
mlir::Value mlirScalarEquals(mlir::Value left, mlir::Value right, mlir::Location loc, mlir::OpBuilder& builder);

// Compare two aggregate MLIR values (LLVM struct-based tuples), recursively
mlir::Value mlirAggregateEquals(mlir::Value left, mlir::Value right, mlir::Location loc, mlir::OpBuilder &builder);
