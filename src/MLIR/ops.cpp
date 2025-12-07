#include "AST.h"
#include "MLIRgen.h"
#include "Types.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "ConstantFolding.h"
#include "CompileTimeExceptions.h"
#include <iostream>

void MLIRGen::visit(ParenExpr* node) {
    node->expr->accept(*this);
}

void MLIRGen::visit(UnaryExpr* node) {
    if (tryEmitConstantForNode(node)) return;
    node->operand->accept(*this);
    VarInfo operand = popValue();
    // Ensure we operate on a scalar: get SSA value (loads memref if needed)
    mlir::Value operandVal = getSSAValue(operand);

    if (node->op == "-") {
        auto zero = builder_.create<mlir::arith::ConstantOp>(
            loc_, operandVal.getType(), builder_.getZeroAttr(operandVal.getType()));
        auto result = builder_.create<mlir::arith::SubIOp>(loc_, zero, operandVal);

        // Store scalar result into a memref-backed VarInfo and push
        VarInfo outVar(operand.type);
        allocaLiteral(&outVar, node->line);
        builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
        outVar.identifier = "";
        pushValue(outVar);
        return;
    }

    // No-op: push original operand through
    operand.identifier = "";
    pushValue(operand);
}

void MLIRGen::visit(AddExpr* node){
    if (tryEmitConstantForNode(node)) return;

    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    node->right->accept(*this);

    // Pop in reverse order
    VarInfo rightInfo = popValue();

    // Determine the promoted type (semantic analysis already set node->type)
    CompleteType promotedType = node->type;
    if (promotedType.baseType == BaseType::UNKNOWN) {
        // Fallback: try to promote manually
        promotedType = promote(leftInfo.type, rightInfo.type);
        if (promotedType.baseType == BaseType::UNKNOWN) {
            promotedType = promote(rightInfo.type, leftInfo.type);
        }
        if (promotedType.baseType == BaseType::UNKNOWN) {
            throw std::runtime_error("AddExpr: cannot promote types for addition");
        }
    }

    // Cast both operands to the promoted type
    VarInfo leftPromoted = castType(&leftInfo, &promotedType, node->line);
    VarInfo rightPromoted = castType(&rightInfo, &promotedType, node->line);

    // Scalar addition/subtraction
    if (promotedType.baseType == BaseType::INTEGER ||
        promotedType.baseType == BaseType::REAL) {
        mlir::Value leftLoaded = getSSAValue(leftPromoted);
        mlir::Value rightLoaded = getSSAValue(rightPromoted);

        mlir::Value result;
        if (promotedType.baseType == BaseType::INTEGER) {
            if (node->op == "+") {
                result = builder_.create<mlir::arith::AddIOp>(loc_, leftLoaded, rightLoaded);
            } else {
                result = builder_.create<mlir::arith::SubIOp>(loc_, leftLoaded, rightLoaded);
            }
        } else {
            if (node->op == "+") {
                result = builder_.create<mlir::arith::AddFOp>(loc_, leftLoaded, rightLoaded);
            } else {
                result = builder_.create<mlir::arith::SubFOp>(loc_, leftLoaded, rightLoaded);
            }
        }

        VarInfo outVar(node->type);
        if (outVar.type.baseType == BaseType::UNKNOWN) {
            throw std::runtime_error("visit(AddExpr*): expression has UNKNOWN type");
        }
        allocaLiteral(&outVar, node->line);
        builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
        outVar.identifier = "";
        pushValue(outVar);
        return;
    }

    // Array / vector / matrix: element-wise addition/subtraction
    if (promotedType.baseType == BaseType::ARRAY ||
        promotedType.baseType == BaseType::VECTOR ||
        promotedType.baseType == BaseType::MATRIX) {

        if (!leftPromoted.value) {
            allocaVar(&leftPromoted, node->line);
        }
        if (!rightPromoted.value) {
            allocaVar(&rightPromoted, node->line);
        }

        if (leftPromoted.runtimeDims.empty()) {
            leftPromoted.runtimeDims = promotedType.dims;
        }
        if (rightPromoted.runtimeDims.empty()) {
            rightPromoted.runtimeDims = promotedType.dims;
        }

        auto &lhsDims = leftPromoted.runtimeDims;
        auto &rhsDims = rightPromoted.runtimeDims;

        if (lhsDims.size() != rhsDims.size() ||
            lhsDims.empty() || rhsDims.empty()) {
            throw SizeError(node->line,
                            "MLIRGen::AddExpr: mismatched ranks for element-wise add/sub");
        }

        // 1D arrays / vectors
        if (lhsDims.size() == 1) {
            int64_t len = lhsDims[0];
            if (len < 0 || rhsDims[0] != len) {
                throw SizeError(node->line,
                                "MLIRGen::AddExpr: mismatched 1D lengths");
            }

            VarInfo outVar(node->type);
            if (promotedType.baseType == BaseType::VECTOR) {
                outVar.runtimeDims = {static_cast<int>(len)};
                mlir::Value vec = allocaVector(static_cast<int>(len), &outVar);
                outVar.value = vec;
            } else {
                allocaLiteral(&outVar, node->line);
            }

            auto idxTy = builder_.getIndexType();
            mlir::Value lb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 0));
            mlir::Value ub = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, len));
            mlir::Value step = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 1));

            auto forOp = builder_.create<mlir::scf::ForOp>(loc_, lb, ub, step, mlir::ValueRange{});

            builder_.setInsertionPointToStart(forOp.getBody());
            {
                mlir::Value idx = forOp.getInductionVar();

                mlir::Value lElem = builder_.create<mlir::memref::LoadOp>(loc_, leftPromoted.value, mlir::ValueRange{idx});
                mlir::Value rElem = builder_.create<mlir::memref::LoadOp>(loc_, rightPromoted.value, mlir::ValueRange{idx});

                mlir::Value elemResult;
                if (lElem.getType().isa<mlir::IntegerType>()) {
                    if (node->op == "+") {
                        elemResult = builder_.create<mlir::arith::AddIOp>(loc_, lElem, rElem);
                    } else {
                        elemResult = builder_.create<mlir::arith::SubIOp>(loc_, lElem, rElem);
                    }
                } else {
                    if (node->op == "+") {
                        elemResult = builder_.create<mlir::arith::AddFOp>(loc_, lElem, rElem);
                    } else {
                        elemResult = builder_.create<mlir::arith::SubFOp>(loc_, lElem, rElem);
                    }
                }

                builder_.create<mlir::memref::StoreOp>(loc_, elemResult, outVar.value, mlir::ValueRange{idx});
            }

            builder_.setInsertionPointAfter(forOp);
            outVar.identifier = "";
            pushValue(outVar);
            return;
        }

        // 2D matrices / 2D arrays
        if (lhsDims.size() == 2) {
            int64_t rows = lhsDims[0];
            int64_t cols = lhsDims[1];
            if (rows < 0 || cols < 0 ||
                rhsDims[0] != rows || rhsDims[1] != cols) {
                throw SizeError(node->line,
                                "MLIRGen::AddExpr: mismatched 2D dimensions");
            }

            VarInfo outVar(node->type);
            allocaLiteral(&outVar, node->line);

            auto idxTy = builder_.getIndexType();
            mlir::Value rowLb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 0));
            mlir::Value rowUb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, rows));
            mlir::Value one = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 1));

            auto outerFor = builder_.create<mlir::scf::ForOp>(loc_, rowLb, rowUb, one, mlir::ValueRange{});

            builder_.setInsertionPointToStart(outerFor.getBody());
            {
                mlir::Value rowIdx = outerFor.getInductionVar();

                mlir::Value colLb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 0));
                mlir::Value colUb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, cols));

                auto innerFor = builder_.create<mlir::scf::ForOp>(loc_, colLb, colUb, one, mlir::ValueRange{});

                builder_.setInsertionPointToStart(innerFor.getBody());
                {
                    mlir::Value colIdx = innerFor.getInductionVar();

                    mlir::Value lElem = builder_.create<mlir::memref::LoadOp>(loc_, leftPromoted.value, mlir::ValueRange{rowIdx, colIdx});
                    mlir::Value rElem = builder_.create<mlir::memref::LoadOp>(loc_, rightPromoted.value, mlir::ValueRange{rowIdx, colIdx});

                    mlir::Value elemResult;
                    if (lElem.getType().isa<mlir::IntegerType>()) {
                        if (node->op == "+") {
                            elemResult = builder_.create<mlir::arith::AddIOp>(loc_, lElem, rElem);
                        } else {
                            elemResult = builder_.create<mlir::arith::SubIOp>(loc_, lElem, rElem);
                        }
                    } else {
                        if (node->op == "+") {
                            elemResult = builder_.create<mlir::arith::AddFOp>(loc_, lElem, rElem);
                        } else {
                            elemResult = builder_.create<mlir::arith::SubFOp>(loc_, lElem, rElem);
                        }
                    }

                    builder_.create<mlir::memref::StoreOp>(loc_, elemResult, outVar.value, mlir::ValueRange{rowIdx, colIdx});
                }

                builder_.setInsertionPointAfter(innerFor);
            }

            builder_.setInsertionPointAfter(outerFor);
            outVar.identifier = "";
            pushValue(outVar);
            return;
        }

        throw std::runtime_error("MLIRGen::AddExpr: Unsupported composite rank for addition: " + toString(promotedType));
    }

    throw std::runtime_error("MLIRGen Error: Unsupported type for addition: " + toString(promotedType));
}


void MLIRGen::visit(ExpExpr* node) {
    if (tryEmitConstantForNode(node)) return;

    // Check if constant folding occurred, but don't skip runtime checks for invalid ops
    bool wasConstant = node->constant.has_value();
    
    if (wasConstant) {
        // Try to emit as constant, but if it fails,
        // fall through to runtime code generation
        try {
            VarInfo lit = createLiteralFromConstant(node->constant.value(), node->type, node->line);
            pushValue(lit);
            return;
        } catch (...) {
            // Constant creation failed - fall through to runtime code
            // This ensures runtime error checks are still generated
        }
    }

    // Generate runtime code with error checking
    node->left->accept(*this);
    VarInfo left = popValue();
    mlir::Value lhs = getSSAValue(left);

    node->right->accept(*this);
    VarInfo right = popValue();
    mlir::Value rhs = getSSAValue(right);

    bool isInt = lhs.getType().isa<mlir::IntegerType>();

    // Promote to float if needed
    auto f32Type = builder_.getF32Type();
    if (lhs.getType().isa<mlir::IntegerType>()) {
        lhs = builder_.create<mlir::arith::SIToFPOp>(loc_, f32Type, lhs);
    }
    if (rhs.getType().isa<mlir::IntegerType>()) {
        rhs = builder_.create<mlir::arith::SIToFPOp>(loc_, f32Type, rhs);
    }

    auto zero = builder_.create<mlir::arith::ConstantOp>(loc_, lhs.getType(), builder_.getZeroAttr(lhs.getType()));
    mlir::Value isBaseZero = builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OEQ, lhs, zero);
    mlir::Value isExpNegative = builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OLT, rhs, zero);
    mlir::Value invalidExp = builder_.create<mlir::arith::AndIOp>(loc_, isBaseZero, isExpNegative);

    auto ifOp = builder_.create<mlir::scf::IfOp>(loc_, lhs.getType(), invalidExp, true);

    // Then region: error
    builder_.setInsertionPointToStart(&ifOp.getThenRegion().front());
    {
        auto mathErrorFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("MathError");
        if (mathErrorFunc) {
            std::string errorMsgStr = "Math Error: 0 cannot be raised to a negative power.";
            std::string errorMsgName = "pow_zero_err_str";
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
             builder_.create<mlir::LLVM::CallOp>(loc_, mathErrorFunc, mlir::ValueRange{errorMsgPtr});
        }
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{zero});
    }

    // Else region: valid exponentiation
    builder_.setInsertionPointToStart(&ifOp.getElseRegion().front());
    {
        mlir::Value powResult = builder_.create<mlir::math::PowFOp>(loc_, lhs, rhs);
        builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{powResult});
    }

    mlir::Value result = ifOp.getResult(0);
    builder_.setInsertionPointAfter(ifOp);

    if (isInt) {
        mlir::Value floored = builder_.create<mlir::math::FloorOp>(loc_, result);
        auto intType = builder_.getI32Type();
        result = builder_.create<mlir::arith::FPToSIOp>(loc_, intType, floored);
    }

    VarInfo outVar(left.type);
    allocaLiteral(&outVar, node->line);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}

void MLIRGen::visit(MultExpr* node){
    if (tryEmitConstantForNode(node)) return;

    node->left->accept(*this);
    VarInfo leftInfo = popValue();
    node->right->accept(*this);
    VarInfo rightInfo = popValue();

    // Determine the promoted type (semantic analysis already set node->type)
    CompleteType promotedType = node->type;
    if (promotedType.baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("MLIRGen::MultExpr: Node type is " + toString(promotedType) + ".");
    }

    // Cast both operands to the promoted type
    VarInfo leftPromoted = castType(&leftInfo, &promotedType, node->line);
    VarInfo rightPromoted = castType(&rightInfo, &promotedType, node->line);

    // Normalize operands to SSA values (loads memref if needed)
    mlir::Value left = getSSAValue(leftPromoted);
    mlir::Value right = getSSAValue(rightPromoted);

    mlir::Value result;

    if (node->op == "/" || node->op == "%") {
        auto zero = builder_.create<mlir::arith::ConstantOp>(loc_, right.getType(), builder_.getZeroAttr(right.getType()));
        mlir::Value isZero;
        if (right.getType().isa<mlir::IntegerType>()) {
            isZero = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::eq, right, zero);
        } else { // FloatType
            isZero = builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OEQ, right, zero);
        }

        auto ifOp = builder_.create<mlir::scf::IfOp>(loc_, left.getType(), isZero, /*hasElse=*/true);

        // --- THEN Region (divisor is zero) ---
        builder_.setInsertionPointToStart(&ifOp.getThenRegion().front());
        {
            auto mathErrorFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("MathError");
            if (mathErrorFunc) {
                 std::string errorMsgStr = "Divide by zero error";
                 std::string errorMsgName = "div_by_zero_err_str";
                 mlir::Value errorMsgPtr;
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
                 errorMsgPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, globalErrorMsg);
                 builder_.create<mlir::LLVM::CallOp>(loc_, mathErrorFunc, mlir::ValueRange{errorMsgPtr});
            }
            builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{zero});
        }

        // --- ELSE Region (divisor is not zero) ---
        builder_.setInsertionPointToStart(&ifOp.getElseRegion().front());
        {
            mlir::Value divResult;
            if (left.getType().isa<mlir::IntegerType>()) {
                if (node->op == "/") divResult = builder_.create<mlir::arith::DivSIOp>(loc_, left, right);
                else divResult = builder_.create<mlir::arith::RemSIOp>(loc_, left, right);
            } else { // FloatType
                divResult = builder_.create<mlir::arith::DivFOp>(loc_, left, right);
            }
            builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{divResult});
        }
        
        result = ifOp.getResult(0);
        builder_.setInsertionPointAfter(ifOp);

    } else { // Multiplication
        if(left.getType().isa<mlir::IntegerType>()) {
            result = builder_.create<mlir::arith::MulIOp>(loc_, left, right);
        } else if(left.getType().isa<mlir::FloatType>()) {
            result = builder_.create<mlir::arith::MulFOp>(loc_, left, right);
        } else {
            throw std::runtime_error("MLIRGen Error: Unsupported type for multiplication.");
        }
    }

    VarInfo outVar(node->type);
    allocaLiteral(&outVar, node->line);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}
void MLIRGen::visit(DotExpr *node){
    // If this DotExpr was folded into a compile-time constant, emit it
    // now and return.
    //TODO add vector support to tryEmitConstantForNode so we can put vectors on the stack
    if (tryEmitConstantForNode(node)) return; // THIS SHOULD WORK - NO NEED TO COMPLICATED IN MLIR


    // Runtime lowering: implement vector (1D) dot product
    // as an scf.for loop that sums element-wise products. Other
    // shapes (matrix-matrix, vector-matrix) currently fall back to
    // emitting an allocated zero result to avoid value-stack underflow.

    //! this is basically not used at all
    // Evaluate operands
    if (node->left) node->left->accept(*this);
    VarInfo leftInfo = popValue();
    if (node->right) node->right->accept(*this);
    VarInfo rightInfo = popValue();

    // Helper to detect 1D containers
    auto is1D = [](const CompleteType &t){
        return t.baseType == BaseType::ARRAY || t.baseType == BaseType::VECTOR;
    };

    if (is1D(leftInfo.type) && is1D(rightInfo.type)) {
        // Ensure we have concrete (compile-time) lengths
        if (leftInfo.type.dims.empty() || rightInfo.type.dims.empty()) {
            throw SizeError(node->line, "MLIRGen: dynamic vector lengths not supported in dot product");
        }
        int64_t Llen = leftInfo.type.dims[0];
        int64_t Rlen = rightInfo.type.dims[0];
        if (Llen < 0 || Rlen < 0) {
            throw SizeError(node->line, "MLIRGen: invalid vector dimensions for dot product");
        }
        if (Llen != Rlen) {
            throw SizeError(node->line, "MLIRGen: vector lengths must match for dot product");
        }

        // Create accumulator (scalar) of the node's result type
        VarInfo acc(node->type);
        allocaLiteral(&acc, node->line);

        // Initialize accumulator to zero
        mlir::Type accSSAType = getLLVMType(node->type);
        mlir::Value zeroConst = builder_.create<mlir::arith::ConstantOp>(loc_, accSSAType, builder_.getZeroAttr(accSSAType));
        builder_.create<mlir::memref::StoreOp>(loc_, zeroConst, acc.value, mlir::ValueRange{});

        // Loop index type
        auto idxTy = builder_.getIndexType();
        mlir::Value lb = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 0));
        mlir::Value ub = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, Llen));
        mlir::Value step = builder_.create<mlir::arith::ConstantOp>(loc_, idxTy, builder_.getIntegerAttr(idxTy, 1));

        auto forOp = builder_.create<mlir::scf::ForOp>(loc_, lb, ub, step, mlir::ValueRange{});

        // Insert loop body
        builder_.setInsertionPointToStart(forOp.getBody());
        {
            mlir::Value idx = forOp.getInductionVar();

            // load elements from each operand
            mlir::Value lElem = builder_.create<mlir::memref::LoadOp>(loc_, leftInfo.value, mlir::ValueRange{idx});
            mlir::Value rElem = builder_.create<mlir::memref::LoadOp>(loc_, rightInfo.value, mlir::ValueRange{idx});

            // Wrap and promote element types to the accumulator type
            if (leftInfo.type.subTypes.empty() || rightInfo.type.subTypes.empty()) {
                throw SizeError(node->line, "MLIRGen: dot product element type unknown");
            }
            CompleteType leftElemCT = leftInfo.type.subTypes[0];
            CompleteType rightElemCT = rightInfo.type.subTypes[0];
            CompleteType targetElemCT = node->type; // scalar promoted type

            VarInfo lElemVar(leftElemCT);
            allocaLiteral(&lElemVar, node->line);
            builder_.create<mlir::memref::StoreOp>(loc_, lElem, lElemVar.value, mlir::ValueRange{});
            VarInfo rElemVar(rightElemCT);
            allocaLiteral(&rElemVar, node->line);
            builder_.create<mlir::memref::StoreOp>(loc_, rElem, rElemVar.value, mlir::ValueRange{});

            VarInfo lProm = promoteType(&lElemVar, &targetElemCT, node->line);
            VarInfo rProm = promoteType(&rElemVar, &targetElemCT, node->line);

            mlir::Value lVal = getSSAValue(lProm);
            mlir::Value rVal = getSSAValue(rProm);

            // multiply
            mlir::Value prod;
            if (lVal.getType().isa<mlir::IntegerType>())
                prod = builder_.create<mlir::arith::MulIOp>(loc_, lVal, rVal);
            else
                prod = builder_.create<mlir::arith::MulFOp>(loc_, lVal, rVal);

            mlir::Value accVal = builder_.create<mlir::memref::LoadOp>(loc_, acc.value, mlir::ValueRange{});
            mlir::Value sum;
            if (accVal.getType().isa<mlir::IntegerType>()) {
                sum = builder_.create<mlir::arith::AddIOp>(loc_, accVal, prod);
            } else {
                sum = builder_.create<mlir::arith::AddFOp>(loc_, accVal, prod);
            }

            builder_.create<mlir::memref::StoreOp>(loc_, sum, acc.value, mlir::ValueRange{});
            builder_.create<mlir::scf::YieldOp>(loc_, mlir::ValueRange{});
        }

        // Move insertion point after loop
        builder_.setInsertionPointAfter(forOp);

        acc.identifier = "";
        pushValue(acc);
        return;
    }

    // Fallback: emit an allocated zero result so callers still get a value
    VarInfo out(node->type);
    allocaLiteral(&out, node->line);
    out.identifier = "";
    pushValue(out);
}
