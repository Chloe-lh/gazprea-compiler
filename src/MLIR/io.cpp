#include "MLIRgen.h"

// Not necessary. For part 2
void MLIRGen::visit(InputStatNode* node) { throw std::runtime_error("InputStatNode not implemented"); }

void MLIRGen::visit(OutputStatNode* node) {
    
    if (!node->expr) {
        return;
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

    // Load the value from its memref if needed. Some visitors may push
    // scalar mlir::Value directly (non-memref), so accept both forms.
    mlir::Value loadedValue;
    if (exprVarInfo.value.getType().isa<mlir::MemRefType>()) {
        loadedValue = builder_.create<mlir::memref::LoadOp>(
        loc_, exprVarInfo.value, mlir::ValueRange{});
    } else {
        loadedValue = exprVarInfo.value;
    }

    // Determine format string name and get format string/printf upfront
    const char* formatStrName = nullptr;
    switch (exprVarInfo.type.baseType) {
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
            throw std::runtime_error("MLIRGen::OutputStat: Unsupported type for printing.");
    }

    // Lookup the format string and printf function (these are module-level symbols)
    auto formatString = module_.lookupSymbol<mlir::LLVM::GlobalOp>(formatStrName);
    auto printfFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    
    if (!formatString || !printfFunc) {
        throw std::runtime_error("MLIRGen::OutputStat: Format string or printf function not found.");
    }
    
    // Get the address of the format string - create this before any value transformations
    mlir::Value formatStringPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, formatString);
    
    // Now transform the value if needed (extensions)
    mlir::Value valueToPrint = loadedValue;
    switch (exprVarInfo.type.baseType) {
        case BaseType::BOOL: {                 
             // map to T/F
            auto i8Ty = builder_.getI8Type();
            auto tVal = builder_.create<mlir::arith::ConstantOp>(
                loc_, i8Ty, builder_.getIntegerAttr(i8Ty, static_cast<int>('T')));
            auto fVal = builder_.create<mlir::arith::ConstantOp>(
                loc_, i8Ty, builder_.getIntegerAttr(i8Ty, static_cast<int>('F')));
            valueToPrint = builder_.create<mlir::arith::SelectOp>(loc_, loadedValue, tVal, fVal);
            // Promote to i32 to satisfy vararg promotion for %c
            valueToPrint = builder_.create<mlir::arith::ExtSIOp>(loc_, builder_.getI32Type(), valueToPrint);
            break;
        }
        case BaseType::REAL:
            // Extend float to f64 for printing
            valueToPrint = builder_.create<mlir::arith::ExtFOp>(
                loc_, builder_.getF64Type(), loadedValue);
            break;
        case BaseType::CHARACTER:
            // Extend character to i32 for printing
            valueToPrint = builder_.create<mlir::arith::ExtSIOp>(
                loc_, builder_.getI32Type(), loadedValue);
            break;
        case BaseType::INTEGER:
            // No extension needed for integer
            break;
        case BaseType::STRING:
            // For non-literal strings we'd expect a pointer; assume loadedValue is already a ptr
            break;
        default:
            break;
    }
    
    // Create the printf call with format string and value
    // Both operands (formatStringPtr and valueToPrint) must dominate this operation
    builder_.create<mlir::LLVM::CallOp>(
        loc_, 
        printfFunc, 
        mlir::ValueRange{formatStringPtr, valueToPrint}
    );
}