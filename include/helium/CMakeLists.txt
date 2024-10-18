add_mlir_interface(ShapeInferenceOpInterface)

add_mlir_dialect(HeliumOps Helium)

add_mlir_doc(HeliumDialect HeliumDialect helium/ -gen-dialect-doc)
add_mlir_doc(HeliumOps HeliumOps Helium/ -gen-op-doc)