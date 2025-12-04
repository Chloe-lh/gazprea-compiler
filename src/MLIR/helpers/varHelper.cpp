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
    if (variable->type.baseType == BaseType::ARRAY){
        // RHS must be an array for whole-array assignment/copy
        if (literal->type.baseType != BaseType::ARRAY) {
            throw AssignError(line, "Cannot assign non-array to array variable '");
        }
            // Ensure both storages exist
        if (!variable->value) allocaVar(variable, line);
        if (!literal->value) allocaVar(literal, line);
        size_t n = 0;
        bool varHas = variable->arraySize.has_value();
        bool litHas = literal->arraySize.has_value();
        if (varHas && litHas) {
            if (variable->arraySize.value() != literal->arraySize.value()) {
                throw AssignError(line, "Array initializer length does not match destination array length");
            }
            n = static_cast<size_t>(variable->arraySize.value());
        } else if (varHas) {
            n = static_cast<size_t>(variable->arraySize.value());
        } else if (litHas) {
            // Destination has dynamic/unknown length but literal has a known length.
            // Use the literal length for the copy and record it on the destination
            // variable so further codegen can allocate a static memref if desired.
            n = static_cast<size_t>(literal->arraySize.value());
            variable->arraySize = literal->arraySize;
        } else {
            throw AssignError(line, "cannot determine array length for assignment");
        }
        auto idxTy = builder_.getIndexType();
        for(size_t t=0; t<n;++t){
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
        return;
    }
    // Struct assignment: copy whole LLVM struct value
    if (variable->type.baseType == BaseType::STRUCT) {
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
            mlir::Type elemTy = getLLVMType(varInfo->type);

            if (varInfo->arraySize.has_value()) {
                int64_t n = varInfo->arraySize.value();
                auto memTy = mlir::MemRefType::get({n}, elemTy);
                varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(loc_, memTy);
            } else {
                //TODO dynamic dimension
                // auto memTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
                // // compute runtime size (lower size expression to IndexType)
                // auto idxTy = builder_.getIndexType();
                
                
            }
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
                mlir::ValueRange{},        // symbol operands
                mlir::IntegerAttr()        // no alignment
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
