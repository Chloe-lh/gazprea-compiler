#include <assert.h>
#include "BackEnd.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

BackEnd::BackEnd() : loc(mlir::UnknownLoc::get(&context)) {
    // Load Dialects.
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::cf::ControlFlowDialect>();
    context.loadDialect<mlir::memref::MemRefDialect>(); 

    // Initialize the MLIR context 
    builder = std::make_shared<mlir::OpBuilder>(&context);
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
    
    // Set DataLayout
    // This string specifies a standard 64-bit architecture layout:
    // e = little endian
    // p:64:64:64 = 64-bit pointers, 64-bit aligned
    // iX:Y:Z = integer type alignment preferences
    const char *dataLayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128";
    module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                    builder->getStringAttr(dataLayout));

    builder->setInsertionPointToStart(module.getBody());

    // Some intial setup to get off the ground 
    setupPrintf();
    createGlobalString("%c\0", "charFormat");
    createGlobalString("%d\0", "intFormat");
    createGlobalString("%g\0", "floatFormat");
    createGlobalString("%s\0", "strFormat");
    createGlobalString("\n\0", "newline");
}

int BackEnd::emitModule() {

    // Create a main function 
    mlir::Type intType = mlir::IntegerType::get(&context, 32);
    auto mainType = mlir::LLVM::LLVMFunctionType::get(intType, {}, false);
    mlir::LLVM::LLVMFuncOp mainFunc = builder->create<mlir::LLVM::LLVMFuncOp>(loc, "main", mainType);
    mlir::Block *entry = mainFunc.addEntryBlock();
    builder->setInsertionPointToStart(entry);

    // Get the integer format string we already created.   
    mlir::LLVM::GlobalOp formatString;
    if (!(formatString = module.lookupSymbol<mlir::LLVM::GlobalOp>("intFormat"))) {
        llvm::errs() << "missing format string!\n";
        return 1;
    }

    // Get the format string and print 415
    mlir::Value formatStringPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, formatString);
    mlir::Value intToPrint = builder->create<mlir::LLVM::ConstantOp>(loc, intType, 415);
    mlir::ValueRange args = {formatStringPtr, intToPrint}; 
    mlir::LLVM::LLVMFuncOp printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    builder->create<mlir::LLVM::CallOp>(loc, printfFunc, args);

    // Return 0
    mlir::Value zero = builder->create<mlir::LLVM::ConstantOp>(loc, intType, builder->getIntegerAttr(intType, 0));
    builder->create<mlir::LLVM::ReturnOp>(builder->getUnknownLoc(), zero);    
    
    module.dump();

    if (mlir::failed(mlir::verify(module))) {
        module.emitError("module failed to verify");
        return 1;
    }
    return 0;
}

int BackEnd::lowerDialects() {
    // Set up the MLIR pass manager to iteratively lower all the Ops
    mlir::PassManager pm(&context);

    // Lower SCF to CF (ControlFlow)
    pm.addPass(mlir::createConvertSCFToCFPass());

    // Lower Arith to LLVM
    pm.addPass(mlir::createArithToLLVMConversionPass());

    // Pre-process MemRefs: Expand Strided Metadata
    // This helps lower complex or 0-D memrefs (scalars) that might otherwise fail in Finalize.
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());

    // Lower MemRef to LLVM (Finalize)
    // This converts memref.alloca/load/store to llvm.alloca/load/store.
    // It relies on the DataLayout attribute we set in the constructor.
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

    // Lower CF to LLVM
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());

    // Lower Func to LLVM
    // Must happen AFTER MemRef lowering so that function bodies contain only
    // LLVM-compatible ops (no high-level memref ops).
    pm.addPass(mlir::createConvertFuncToLLVMPass());

    // Cleanup unrealized casts inserted by the conversion passes
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    // Run the passes
    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "Pass pipeline failed\n";
        return 1;
    }
    return 0;
}

void BackEnd::dumpMLIR(std::ostream &os) {
    // Dump the MLIR module to the output stream
    llvm::raw_os_ostream output(os);
    module.print(output);
}

void BackEnd::dumpLLVM(std::ostream &os) {  
    // The only remaining dialects in our module after the passes are builtin
    // and LLVM. Setup translation patterns to get them to LLVM IR.
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);
    llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);

    if (!llvm_module) {
        llvm::errs() << "Failed to translate MLIR module to LLVM IR\n";
        return;
    }

    // Create llvm ostream and dump into the output file
    llvm::raw_os_ostream output(os);
    output << *llvm_module;
}

void BackEnd::setupPrintf() {
    // Create a function declaration for printf, the signature is:
    //   * `i32 (ptr, ...)`
    mlir::Type intType = mlir::IntegerType::get(&context, 32);
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context);
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(intType, ptrTy,
                                                        /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    builder->create<mlir::LLVM::LLVMFuncOp>(loc, "printf", llvmFnType);
}

void BackEnd::createGlobalString(const char *str, const char *stringName) {

    mlir::Type charType = mlir::IntegerType::get(&context, 8);

    // create string and string type
    auto mlirString = mlir::StringRef(str, strlen(str) + 1);
    auto mlirStringType = mlir::LLVM::LLVMArrayType::get(charType, mlirString.size());

    builder->create<mlir::LLVM::GlobalOp>(loc, mlirStringType, /*isConstant=*/true,
                            mlir::LLVM::Linkage::Internal, stringName,
                            builder->getStringAttr(mlirString), /*alignment=*/0);
    return;
}