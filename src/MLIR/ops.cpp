#include "MLIRgen.h"
#include "mlir/Dialect/Math/IR/Math.h"

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
        allocaLiteral(&outVar);
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
    VarInfo leftPromoted = castType(&leftInfo, &promotedType);
    VarInfo rightPromoted = castType(&rightInfo, &promotedType);

    // Load the promoted values (getSSAValue handles memref vs SSA)
    mlir::Value leftLoaded = getSSAValue(leftPromoted);
    mlir::Value rightLoaded = getSSAValue(rightPromoted);

    mlir::Value result;
    if (promotedType.baseType == BaseType::INTEGER) {
        if (node->op == "+") {
            result = builder_.create<mlir::arith::AddIOp>(loc_, leftLoaded, rightLoaded);
        } else if (node->op == "-") {
            result = builder_.create<mlir::arith::SubIOp>(loc_, leftLoaded, rightLoaded);
        }
    } else if (promotedType.baseType == BaseType::REAL) {
        if (node->op == "+") {
            result = builder_.create<mlir::arith::AddFOp>(loc_, leftLoaded, rightLoaded);
        } else if (node->op == "-") {
            result = builder_.create<mlir::arith::SubFOp>(loc_, leftLoaded, rightLoaded);
        }
    } else {
        throw std::runtime_error("MLIRGen Error: Unsupported type for addition: " + toString(promotedType));
    }

    // Use the expression's own type (node->type)
    VarInfo outVar(node->type);
    if (outVar.type.baseType == BaseType::UNKNOWN) {
        throw std::runtime_error("visit(AddExpr*): expression has UNKNOWN type");
    }
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}


void MLIRGen::visit(ExpExpr* node) {
    if (tryEmitConstantForNode(node)) return;

    // Check if constant folding occurred, but don't skip runtime checks for invalid ops
    bool wasConstant = node->constant.has_value();
    
    if (wasConstant) {
        // Try to emit as constant, but if it fails,
        // fall through to runtime code generation
        try {
            VarInfo lit = createLiteralFromConstant(node->constant.value(), node->type);
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
    allocaLiteral(&outVar);
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

    // Normalize operands to SSA values (loads memref if needed)
    mlir::Value left = getSSAValue(leftInfo);
    mlir::Value right = getSSAValue(rightInfo);

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

    VarInfo outVar(leftInfo.type);
    allocaLiteral(&outVar);
    builder_.create<mlir::memref::StoreOp>(loc_, result, outVar.value, mlir::ValueRange{});
    outVar.identifier = "";
    pushValue(outVar);
}