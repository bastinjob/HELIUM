#ifndef MLIR_HELIUM_PASSES_H
#define MLIR_HELIUM_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace Helium {
  std::unique_ptr<mlir::Pass> createLowerToAffinePass();
  std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
}

#endif 