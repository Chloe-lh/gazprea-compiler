#include "CompileTimeExceptions.h"
#include "MLIRgen.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

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
// Declarations / Globals helpers
mlir::Type MLIRGen::getLLVMType(const CompleteType& type) {
    switch (type.baseType) {
        case BaseType::BOOL: return builder_.getI1Type();
        case BaseType::CHARACTER: return builder_.getI8Type();
        case BaseType::INTEGER: return builder_.getI32Type();
        case BaseType::REAL: return builder_.getF32Type();
        case BaseType::TUPLE: {
            if (type.subTypes.empty()) {
                throw std::runtime_error("getLLVMType: empty tuple type is invalid");
            }
            llvm::SmallVector<mlir::Type, 4> elemTys;
            elemTys.reserve(type.subTypes.size());
            for (const auto &st : type.subTypes) {
                elemTys.push_back(getLLVMType(st));
            }
            return mlir::LLVM::LLVMStructType::getLiteral(&context_, elemTys);
        }
        default:
            throw std::runtime_error("getLLVMType: Unsupported type: " + toString(type));
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
    }

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
}

void MLIRGen::allocaLiteral(VarInfo* varInfo, int line) {
    varInfo->isConst = true;
    allocaVar(varInfo, line);
}

bool MLIRGen::tryEmitConstantForNode(ExprNode* node) {
    if (!node) return false;
    if (!node->constant.has_value()) return false;
    try {
        VarInfo lit = createLiteralFromConstant(node->constant.value(), node->type, node->line);
        pushValue(lit);
        return true;
    } catch (...) {
        // unsupported constant type or codegen error; fall back to normal lowering
        return false;
    }
}

void MLIRGen::allocaVar(VarInfo* varInfo, int line) {
    mlir::Block *block = builder_.getBlock();
    if (!block) block = builder_.getInsertionBlock();
    if (!block) {
        throw std::runtime_error("allocaVar: builder has no current block");
    }

    // Find the parent function op from the current block
    mlir::Operation *op = block->getParentOp();
    mlir::func::FuncOp funcOp = nullptr;

    while (op) {
        if (auto f = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
            funcOp = f;
            break;
        }
        op = op->getParentOp();
    }

    if (!funcOp) {
        throw std::runtime_error("allocaVar: could not find parent function for allocation");
    }

    // Always allocate at the beginning of the entry block
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
            {
                auto zero = entryBuilder.create<mlir::arith::ConstantOp>(
                    loc_, builder_.getI1Type(), builder_.getIntegerAttr(builder_.getI1Type(), 0));
                entryBuilder.create<mlir::memref::StoreOp>(loc_, zero, varInfo->value, mlir::ValueRange{});
            }
            break;
        case BaseType::INTEGER:
            varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getI32Type()));
            {
                auto zero = entryBuilder.create<mlir::arith::ConstantOp>(
                    loc_, builder_.getI1Type(), builder_.getIntegerAttr(builder_.getI1Type(), 0));
                entryBuilder.create<mlir::memref::StoreOp>(loc_, zero, varInfo->value, mlir::ValueRange{});
            }
            break;
        case BaseType::REAL:
            varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(
                loc_, mlir::MemRefType::get({}, builder_.getF32Type()));
            {
                auto zero = entryBuilder.create<mlir::arith::ConstantOp>(
                    loc_, builder_.getI1Type(), builder_.getIntegerAttr(builder_.getI1Type(), 0));
                entryBuilder.create<mlir::memref::StoreOp>(loc_, zero, varInfo->value, mlir::ValueRange{});
            }
            
            break;

        case BaseType::TUPLE: {
            if (varInfo->type.subTypes.size() < 2) {
                throw SizeError(line, "Error: Tuple must have at least 2 elements.");
            }
            mlir::Type structTy = getLLVMType(varInfo->type);
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
            auto i64Ty = builder_.getI64Type();
            auto oneAttr = builder_.getIntegerAttr(i64Ty, 1);
            mlir::Value one = entryBuilder.create<mlir::arith::ConstantOp>(loc_, i64Ty, oneAttr);
            varInfo->value = entryBuilder.create<mlir::LLVM::AllocaOp>(loc_, ptrTy, structTy, one, 0);
            break;
        }

        default: {
            std::string varName = varInfo->identifier.empty() ? "<temporary>" : varInfo->identifier;
            throw std::runtime_error("allocaVar: unsupported type " +
                                     std::to_string(static_cast<int>(varInfo->type.baseType)) +
                                     " for variable '" + varName + "'");
        }
    }
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