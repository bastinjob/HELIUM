add_mlir_dialect_library(MLIRHelium
                        HeliumDialect.cpp
                        HeliumOps.cpp
                        LowerToAffine.cpp
                        LowerToLLVM.cpp
                        ShapeInferenceOpInterface.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Helium 

        DEPENDS
        MLIRHeliumOpsIncGen
        MLIRHeliumShapeInferenceOpInterfaceIncGen

        LINK_LIBS PUBLIC  
        MLIRIR  
        )