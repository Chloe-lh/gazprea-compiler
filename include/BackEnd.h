// Ensure single inclusion
#pragma once

// Standard library
#include <memory>
#include <ostream>

// Pass manager
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"

// Translation
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/raw_os_ostream.h"

// MLIR IR
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

// Dialects 
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

class BackEnd {
 public:
    BackEnd();

    int emitModule();
    int lowerDialects();
    void dumpLLVM(std::ostream &os);

   mlir::Location getLoc() const {
      return loc;
   }
   std::shared_ptr<mlir::OpBuilder> getBuilder() const {
      return builder;
   }
   mlir::ModuleOp getModule() const {
      return module;
   }
   mlir::MLIRContext &getContext() {
      return context;
   }
 
 protected:
    void setupPrintf();
    void createGlobalString(const char *str, const char *stringName);
      
 private:
    // MLIR
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    std::shared_ptr<mlir::OpBuilder> builder;
    mlir::Location loc;

    // LLVM 
    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> llvm_module;
};
