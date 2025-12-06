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
        if (literal->type.baseType != BaseType::ARRAY) {
            throw AssignError(line, "Cannot assign non-array to array variable");
            // TODO: Implement int -> array
        }
        // Ensure literal storage exists first (needed to get size for dynamic arrays)
        // But skip allocation for slice structs - they're already values, not memrefs
        if (!literal->value) {
            allocaVar(literal, line);
        } else if (literal->value.getType().isa<mlir::LLVM::LLVMStructType>()) {
            // Slice struct - no allocation needed, it's already a value
        } else {
            // Memref - ensure it's allocated (should already be, but check anyway)
            if (!literal->value) {
                allocaVar(literal, line);
            }
        }

        auto getLen = [](const CompleteType &t) -> std::optional<int64_t> {
            if (t.baseType != BaseType::ARRAY) return std::nullopt;
            if (t.dims.size() != 1) return std::nullopt; // only 1D for now
            if (t.dims[0] < 0) return std::nullopt;
            return static_cast<int64_t>(t.dims[0]);
        };

        std::optional<int64_t> varLen = getLen(variable->type);
        std::optional<int64_t> litLen = getLen(literal->type);

        auto idxTy = builder_.getIndexType();
        mlir::Value nVal; // Runtime size value
        
        if (varLen && litLen) {
            if (varLen.value() != litLen.value()) {
                throw AssignError(line, "Array initializer length does not match destination array length");
            }
            nVal = builder_.create<mlir::arith::ConstantIndexOp>(
                loc_, static_cast<int64_t>(varLen.value()));
        } else if (varLen) {
            nVal = builder_.create<mlir::arith::ConstantIndexOp>(
                loc_, static_cast<int64_t>(varLen.value()));
        } else if (litLen) {
            // Destination had inferred size; adopt the literal's concrete size.
            nVal = builder_.create<mlir::arith::ConstantIndexOp>(
                loc_, static_cast<int64_t>(litLen.value()));
            variable->type.dims = literal->type.dims;
        } else {
            // Both are dynamic - check if source is a slice struct or memref
            if (literal->value.getType().isa<mlir::LLVM::LLVMStructType>()) {
                // Source is a slice struct - extract length from struct
                llvm::SmallVector<int64_t, 1> lenPos{1}; // Second field is length
                mlir::Value sliceLen_i64 = builder_.create<mlir::LLVM::ExtractValueOp>(
                    loc_, literal->value, lenPos
                );
                // Convert i64 to index
                nVal = builder_.create<mlir::arith::IndexCastOp>(
                    loc_, idxTy, sliceLen_i64
                );
                // Update variable type to match (dynamic size)
                variable->type.dims = literal->type.dims;
            } else if (literal->value.getType().isa<mlir::MemRefType>()) {
                // Source is a memref - get size from memref at runtime
                auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
                nVal = builder_.create<mlir::memref::DimOp>(loc_, literal->value, zeroIdx);
                // Update variable type to match (dynamic size)
                variable->type.dims = literal->type.dims;
            } else {
                throw AssignError(line, "cannot determine array length for assignment");
            }
        }
        
        // Ensure destination is allocated (should already be done for dynamic arrays in visit(ArrayTypedDecNode*))
        if (!variable->value) {
            // If variable is dynamic, we need to allocate it with the runtime size
            if (!varLen) {
                // This should not happen if visit(ArrayTypedDecNode*) worked correctly
                throw AssignError(line, "Dynamic array destination not allocated before assignment");
            } else {
                allocaVar(variable, line);
            }
        }
        
        // Create a loop to copy elements
        auto c0 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        auto c1 = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
        
        // Check if source is a slice struct or memref
        bool isSliceStruct = literal->value.getType().isa<mlir::LLVM::LLVMStructType>();
        mlir::Value slicePtr;
        
        if (isSliceStruct) {
            // Extract pointer from slice struct (first field)
            llvm::SmallVector<int64_t, 1> ptrPos{0}; // First field is pointer
            slicePtr = builder_.create<mlir::LLVM::ExtractValueOp>(
                loc_, literal->value, ptrPos
            );
        }
        
        builder_.create<mlir::scf::ForOp>(
            loc_, c0, nVal, c1, mlir::ValueRange{},
            [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value iv, mlir::ValueRange args) {
                // Use loop index iv
                mlir::Value idx = iv;

                // Load source element
                mlir::Value srcElemVal;
                if (isSliceStruct) {
                    // Load from slice pointer using GEP and load
                    auto i32Ty = b.getI32Type();
                    auto i32PtrTy = mlir::LLVM::LLVMPointerType::get(&context_);
                    auto i64Ty = b.getI64Type();
                    auto idx_i64 = b.create<mlir::arith::IndexCastOp>(l, i64Ty, idx);
                    mlir::Value gep = b.create<mlir::LLVM::GEPOp>(
                        l, i32PtrTy, i32Ty, slicePtr, mlir::ValueRange{idx_i64}
                    );
                    srcElemVal = b.create<mlir::LLVM::LoadOp>(l, i32Ty, gep);
                } else {
                    // Load from memref
                    srcElemVal = b.create<mlir::memref::LoadOp>(l, literal->value, mlir::ValueRange{idx});
                }

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
                b.create<mlir::memref::StoreOp>(l, storeVal, variable->value, mlir::ValueRange{idx});
                
                b.create<mlir::scf::YieldOp>(l);
            }
        );
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
        auto memrefType = source->value.getType().cast<mlir::MemRefType>();
        auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        return builder_.create<mlir::memref::DimOp>(loc_, source->value, zeroIdx);
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
    
    throw std::runtime_error("computeArraySize: source is not a memref, slice struct, or array type");
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

void MLIRGen::allocaVar(VarInfo* varInfo, int line, mlir::Value sizeValue) {
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

    // For static allocations, always allocate at the beginning of the entry block
    // For dynamic allocations (with sizeValue), allocate in current block to ensure sizeValue dominates
    mlir::Block &entry = funcOp.front();
    mlir::OpBuilder entryBuilder(&entry, entry.begin());
    
    // Determine if we should allocate in entry block or current block
    bool allocateInEntry = !sizeValue; // Static allocations go in entry block

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
                // Static array size
                int64_t n = varInfo->type.dims[0];
                auto memTy = mlir::MemRefType::get({n}, elemTy);
                varInfo->value = entryBuilder.create<mlir::memref::AllocaOp>(loc_, memTy);
            } else if (varInfo->type.dims.size() == 1 && varInfo->type.dims[0] < 0) {
                // Dynamic array size - requires sizeValue parameter
                if (!sizeValue) {
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
                    mlir::ValueRange{sizeValue}, // dynamic size operand
                    mlir::ValueRange{},           // symbol operands
                    mlir::IntegerAttr()           // no alignment
                );
            } else {
                throw std::runtime_error("allocaVar: unsupported array shape for variable '" +
                                         (varInfo->identifier.empty() ? std::string("<temporary>") : varInfo->identifier) + "'");
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
