#ifndef HELIUM_DIALECT
#define HELIUM_DIALECT

include "mlir/IR/OpBase.td"

def Helium_Dialect : Dialect {

    let name = "Helium";

    let summary = "Helium is Tensor compiler built using MLIR and LLVM";

    let description = [{" Heliumm DIalect has tensor operation like const_op, add_op etc, all designed to be printable"}];


    let cppNamespace = "::Helium";

    let hasConstantMaterializer = 1;
}


//Base helium operation definition
class Helium_Op<string mnemonic, list<Trait> traits = []>:Op<HELIUM_DIALECT, mnemonic, traits>;


#endif 