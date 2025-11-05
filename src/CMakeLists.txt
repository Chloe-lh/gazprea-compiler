# Gather our source files in this directory.
set(
  gazprea_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/BackEnd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/AST.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ASTBuilder.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MLIRgen.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/SemanticAnalysisVisitor.cpp"
    "${CMAKE_SOURCE_DIR}/src/ASTBuilderUtils.cpp"
    "${CMAKE_SOURCE_DIR}/src/Scope.cpp"
    "${CMAKE_SOURCE_DIR}/src/Types.cpp"
)

# Build our executable from the source files.
add_executable(gazc ${gazprea_src_files})
target_include_directories(gazc PUBLIC ${ANTLR_GEN_DIR})
# Make runtime headers (runtime/include) available to the main target so
# files that include run_time_errors.h can find it.
target_include_directories(gazc PUBLIC ${CMAKE_SOURCE_DIR}/runtime/include)

# Ensure that the antlr4-runtime is available.
add_dependencies(gazc antlr)

# Find the libraries that correspond to the LLVM components
# that we wish to use
set(LLVM_LINK_COMPONENTS Core Support)
llvm_map_components_to_libnames(llvm_libs core)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Add the MLIR, LLVM, antlr runtime and parser as libraries to link.
target_link_libraries(gazc PRIVATE
    parser
    antlr4-runtime
    ${llvm_libs}
    ${dialect_libs}
	MLIRLLVMCommonConversion
	MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
)

# Symbolic link our executable to the base directory so we don't have to go searching for it.
symlink_to_bin("gazc")
