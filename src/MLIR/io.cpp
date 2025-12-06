#include "CompileTimeExceptions.h"
#include "MLIRgen.h"
#include "Types.h"

void MLIRGen::emitPrintScalar(const CompleteType &type, mlir::Value value) {
    const char* formatStrName = nullptr;
    switch (type.baseType) {
        case BaseType::BOOL:
            formatStrName = "charFormat";
            break;
        case BaseType::INTEGER:
            formatStrName = "intFormat";
            break;
        case BaseType::REAL:
            formatStrName = "floatFormat";
            break;
        case BaseType::CHARACTER:
            formatStrName = "charFormat";
            break;
        case BaseType::STRING:
            formatStrName = "strFormat";
            break;
        default:
            throw std::runtime_error("MLIRGen::emitPrintScalar: Unsupported type for printing.");
    }

    auto formatString = module_.lookupSymbol<mlir::LLVM::GlobalOp>(formatStrName);
    auto printfFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    if (!formatString || !printfFunc) {
        throw std::runtime_error("MLIRGen::emitPrintScalar: Format string or printf not found.");
    }

    mlir::Value formatStringPtr =
        builder_.create<mlir::LLVM::AddressOfOp>(loc_, formatString);

    mlir::Value valueToPrint = value;
    switch (type.baseType) {
        case BaseType::BOOL: {
            auto i8Ty = builder_.getI8Type();
            auto tVal = builder_.create<mlir::arith::ConstantOp>(
                loc_, i8Ty, builder_.getIntegerAttr(i8Ty, static_cast<int>('T')));
            auto fVal = builder_.create<mlir::arith::ConstantOp>(
                loc_, i8Ty, builder_.getIntegerAttr(i8Ty, static_cast<int>('F')));
            valueToPrint = builder_.create<mlir::arith::SelectOp>(loc_, value, tVal, fVal);
            valueToPrint = builder_.create<mlir::arith::ExtSIOp>(
                loc_, builder_.getI32Type(), valueToPrint);
            break;
        }
        case BaseType::REAL:
            valueToPrint = builder_.create<mlir::arith::ExtFOp>(
                loc_, builder_.getF64Type(), valueToPrint);
            break;
        case BaseType::CHARACTER:
            valueToPrint = builder_.create<mlir::arith::ExtSIOp>(
                loc_, builder_.getI32Type(), valueToPrint);
            break;
        case BaseType::INTEGER:
            break;
        case BaseType::STRING:
            // assume valueToPrint is already a pointer
            break;
        default:
            break;
    }

    builder_.create<mlir::LLVM::CallOp>(
        loc_,
        printfFunc,
        mlir::ValueRange{formatStringPtr, valueToPrint});
}
void MLIRGen::emitPrintMatrix(const VarInfo &matrixVarInfo) {
    if (matrixVarInfo.type.baseType != BaseType::MATRIX) {
        throw std::runtime_error("emitPrintMatrix: non-matrix type");
    }
    if (!matrixVarInfo.value || !matrixVarInfo.value.getType().isa<mlir::MemRefType>()) {
        throw std::runtime_error("emitPrintMatrix: matrix has no memref storage");
    }
    if (matrixVarInfo.type.dims.size() != 2 || matrixVarInfo.type.dims[0] < 0) {
        throw std::runtime_error("emitPrintMatrix: only static 2D arrays supported for printing");
    }

    int64_t rows = matrixVarInfo.type.dims[0];
    int64_t cols = matrixVarInfo.type.dims[1];
    CompleteType elemType = matrixVarInfo.type.subTypes.empty()
                                ? CompleteType(BaseType::UNKNOWN)
                                : matrixVarInfo.type.subTypes[0];

    auto idxTy = builder_.getIndexType();
    auto i8Ty = builder_.getI8Type();

    // Print outer '['
    {
        auto c = builder_.create<mlir::arith::ConstantOp>(
            loc_, i8Ty,
            builder_.getIntegerAttr(i8Ty, static_cast<int>('['))
        );
        emitPrintScalar(CompleteType(BaseType::CHARACTER), c.getResult());
    }

    // Print each row
    for (int64_t r = 0; r < rows; ++r) {
        // Print inner '['
        {
            auto c = builder_.create<mlir::arith::ConstantOp>(
                loc_, i8Ty,
                builder_.getIntegerAttr(i8Ty, static_cast<int>('['))
            );
            emitPrintScalar(CompleteType(BaseType::CHARACTER), c.getResult());
        }

        // Print row elements
        for (int64_t col = 0; col < cols; ++col) {
            // Print space before all but first element
            if (col > 0) {
                auto space = builder_.create<mlir::arith::ConstantOp>(
                    loc_, i8Ty,
                    builder_.getIntegerAttr(i8Ty, static_cast<int>(' '))
                );
                emitPrintScalar(CompleteType(BaseType::CHARACTER), space.getResult());
            }

            auto rConst = builder_.create<mlir::arith::ConstantOp>(
                loc_, idxTy, builder_.getIntegerAttr(idxTy, r)
            );
            auto cConst = builder_.create<mlir::arith::ConstantOp>(
                loc_, idxTy, builder_.getIntegerAttr(idxTy, col)
            );

            auto elem = builder_.create<mlir::memref::LoadOp>(
                loc_, matrixVarInfo.value, mlir::ValueRange{rConst, cConst}
            );

            emitPrintScalar(elemType, elem.getResult());
        }

        // Print inner ']'
        {
            auto c = builder_.create<mlir::arith::ConstantOp>(
                loc_, i8Ty,
                builder_.getIntegerAttr(i8Ty, static_cast<int>(']'))
            );
            emitPrintScalar(CompleteType(BaseType::CHARACTER), c.getResult());
        }

        // Print space between rows (except after last row)
        if (r < rows - 1) {
            auto space = builder_.create<mlir::arith::ConstantOp>(
                loc_, i8Ty,
                builder_.getIntegerAttr(i8Ty, static_cast<int>(' '))
            );
            emitPrintScalar(CompleteType(BaseType::CHARACTER), space.getResult());
        }
    }

    // Print outer ']'
    {
        auto c = builder_.create<mlir::arith::ConstantOp>(
            loc_, i8Ty,
            builder_.getIntegerAttr(i8Ty, static_cast<int>(']'))
        );
        emitPrintScalar(CompleteType(BaseType::CHARACTER), c.getResult());
    }
}

void MLIRGen::emitPrintArray(const VarInfo &arrayVarInfo) {
    if (arrayVarInfo.type.baseType != BaseType::ARRAY) {
        throw std::runtime_error("emitPrintArray: non-array type");
    }
    if (!arrayVarInfo.value ||
        !arrayVarInfo.value.getType().isa<mlir::MemRefType>()) {
        throw std::runtime_error("emitPrintArray: array has no memref storage");
    }
    if (arrayVarInfo.type.dims.size() != 1 || arrayVarInfo.type.dims[0] < 0) {
        throw std::runtime_error("emitPrintArray: only static 1D arrays supported for printing");
    }

    int64_t n = arrayVarInfo.type.dims[0];
    CompleteType elemType = arrayVarInfo.type.subTypes.empty()
                                ? CompleteType(BaseType::UNKNOWN)
                                : arrayVarInfo.type.subTypes[0];

    // Print '['
    {
        auto i8Ty = builder_.getI8Type();
        auto c = builder_.create<mlir::arith::ConstantOp>(
            loc_, i8Ty,
            builder_.getIntegerAttr(i8Ty, static_cast<int> ('[')));
        emitPrintScalar(CompleteType(BaseType::CHARACTER), c.getResult());
    }

    auto idxTy = builder_.getIndexType();
    for (int64_t i = 0; i < n; ++i) {
        // Print space before all but the first element
        if (i > 0) {
            auto i8Ty = builder_.getI8Type();
            auto space = builder_.create<mlir::arith::ConstantOp>(
                loc_, i8Ty,
                builder_.getIntegerAttr(i8Ty, static_cast<int>(' ')));
            emitPrintScalar(CompleteType(BaseType::CHARACTER), space.getResult());
        }

        auto idxConst = builder_.create<mlir::arith::ConstantOp>(
            loc_, idxTy,
            builder_.getIntegerAttr(idxTy, static_cast<int64_t>(i)));
        mlir::Value elemVal = builder_.create<mlir::memref::LoadOp>(
            loc_, arrayVarInfo.value, mlir::ValueRange{idxConst});
        emitPrintScalar(elemType, elemVal);
    }

    // Print ']'
    {
        auto i8Ty = builder_.getI8Type();
        auto c = builder_.create<mlir::arith::ConstantOp>(
            loc_, i8Ty,
            builder_.getIntegerAttr(i8Ty, static_cast<int> (']')));
        emitPrintScalar(CompleteType(BaseType::CHARACTER), c.getResult());
    }
}

void MLIRGen::visit(InputStatNode* node) {
    // Resolve the variable to read into
    VarInfo* targetVar = currScope_->resolveVar(node->name, node->line);
    if (!targetVar) {
        throw SymbolError(node->line, "InputStat: variable '" + node->name + "' not found.");
    }
    
    if (targetVar->isConst) {
        throw AssignError(node->line, "InputStat: cannot read into const variable '" + node->name + "'.");
    }
    
    // Ensure the variable has storage allocated
    if (!targetVar->value) {
        allocaVar(targetVar, node->line);
    }
    
    // Get a pointer to the variable for the runtime function
    mlir::Value varPtr;
    if (targetVar->value.getType().isa<mlir::MemRefType>()) {
        // For memref variables, extract the base pointer and convert to LLVM pointer
        auto memrefType = targetVar->value.getType().cast<mlir::MemRefType>();
        auto elemType = memrefType.getElementType();
        auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context_);
        
        // Extract the base pointer from the memref descriptor
        // This gets the actual pointer value from the memref
        mlir::Value baseIndex = builder_.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
            loc_, targetVar->value
        );
        
        // Convert the index to an LLVM pointer
        // Use IndexCastOp to convert index to pointer-sized integer, then cast to pointer
        auto indexTy = builder_.getIndexType();
        auto i64Ty = builder_.getI64Type();
        mlir::Value ptrInt = builder_.create<mlir::arith::IndexCastOp>(
            loc_, i64Ty, baseIndex
        );
        
        // Convert integer to LLVM pointer
        varPtr = builder_.create<mlir::LLVM::IntToPtrOp>(loc_, ptrTy, ptrInt);
    } else {
        // Already an LLVM pointer (e.g., for tuples)
        varPtr = targetVar->value;
    }
    
    // Determine which runtime function to call based on type
    mlir::LLVM::LLVMFuncOp readFunc = nullptr;
    switch (targetVar->type.baseType) {
        case BaseType::INTEGER:
            readFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("readInt");
            break;
        case BaseType::REAL:
            readFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("readReal");
            break;
        case BaseType::CHARACTER:
            readFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("readChar");
            break;
        case BaseType::BOOL:
            readFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("readBool");
            break;
        default:
            throw std::runtime_error("InputStat: unsupported type for input: " + toString(targetVar->type));
    }
    
    if (!readFunc) {
        throw std::runtime_error("InputStat: runtime function not found for type: " + toString(targetVar->type));
    }
    
    // Call the runtime function with the variable pointer
    builder_.create<mlir::LLVM::CallOp>(loc_, readFunc, mlir::ValueRange{varPtr});
}

void MLIRGen::visit(OutputStatNode* node) {
    if (!node->expr) {
        throw std::runtime_error("MLIRGen::OutpuStatNode: No expr found");
    }

    // Handle string literals
    if (auto strNode = std::dynamic_pointer_cast<StringNode>(node->expr)) {
        auto printfFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
        auto formatString = module_.lookupSymbol<mlir::LLVM::GlobalOp>("strFormat");
        if (!printfFunc || !formatString) {
            throw std::runtime_error("MLIRGen::OutputStat: missing printf or strFormat.");
        }
        // Create/find a global for the string literal in curr scope
        std::string symName = std::string("strlit_") + std::to_string(std::hash<std::string>{}(strNode->value));
        auto existing = module_.lookupSymbol<mlir::LLVM::GlobalOp>(symName);
        if (!existing) {
            auto* moduleBuilder = backend_.getBuilder().get();
            auto savedIP = moduleBuilder->saveInsertionPoint();
            moduleBuilder->setInsertionPointToStart(module_.getBody());
            mlir::Type charTy = builder_.getI8Type();
            mlir::StringRef sref(strNode->value.c_str(), strNode->value.size() + 1);
            auto arrTy = mlir::LLVM::LLVMArrayType::get(charTy, sref.size());
            moduleBuilder->create<mlir::LLVM::GlobalOp>(loc_, arrTy, /*constant=*/true,
                mlir::LLVM::Linkage::Internal, symName, builder_.getStringAttr(sref), 0);
            moduleBuilder->restoreInsertionPoint(savedIP);
            existing = module_.lookupSymbol<mlir::LLVM::GlobalOp>(symName);
        }
        auto fmtPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, formatString);
        auto strPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, existing);
        builder_.create<mlir::LLVM::CallOp>(loc_, printfFunc, mlir::ValueRange{fmtPtr, strPtr});
        return;
    }

    // Evaluate the expression to get the value to print
    node->expr->accept(*this);
    VarInfo exprVarInfo = popValue();
    std::cerr << "exprVarInfo type: " << toString(exprVarInfo.type.baseType);
    if (exprVarInfo.type.baseType == BaseType::MATRIX){
        emitPrintMatrix(exprVarInfo);
        return;
    }else if (exprVarInfo.type.baseType == BaseType::ARRAY) {
        emitPrintArray(exprVarInfo);
        return;
    } else if (exprVarInfo.type.baseType == BaseType::VECTOR) {
        throw TypeError(node->line, "Cannot print vectors");
    } else if (isScalarType(exprVarInfo.type.baseType)) {
        mlir::Value loadedValue = getSSAValue(exprVarInfo);
        emitPrintScalar(exprVarInfo.type, loadedValue);
    } else {
        throw std::runtime_error("MLIRGen::OutputStat: unsupported type for printing");
    }
}
