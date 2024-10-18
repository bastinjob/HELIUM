get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRStarK
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRLLVMDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIROptLib
        )
add_llvm_executable(helium-opt helium-opt.cpp)

llvm_update_compile_flags(helium-opt)
target_link_libraries(helium-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(helium-opt)