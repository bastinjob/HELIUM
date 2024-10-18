#include "helium/HeliumDialect.h"
#include "helium/HeliumOps.h"
#include "helium/HeliumPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

static mlir::MemRefType convertTensorToMemRef(mlir::TensorType type){
    assert(type.hasRank() && "expected only ranked shapes");
    return mlir::MemRefType::get(type.getShape(), type.getElementType());
}
static mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc,
                                         mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

class ConstantOpLowering : public mlir::OpRewritePattern<Helium::ConstantOp> {
  using OpRewritePattern<Helium::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(Helium::ConstantOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::DenseElementsAttr constantValue = op.getValue();
    mlir::Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<mlir::TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    mlir::SmallVector<mlir::Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
          0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
    }


    mlir::SmallVector<mlir::Value, 2> indices;
    auto valueIt = constantValue.getValues<mlir::FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::AffineStoreOp>(
            loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};

class PrintOpLowering : public mlir::OpConversionPattern<Helium::PrintOp> {
  using OpConversionPattern<Helium::PrintOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(Helium::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
      // We don't lower "Helium.print" in this pass, but we need to update its
      // operands.
      rewriter.updateRootInPlace(op,
                                 [&] { op->setOperands(adaptor.getOperands()); });
      return mlir::success();
  }
};

namespace {
class HeliumToAffineLowerPass : public mlir::PassWrapper<HeliumToAffineLowerPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HeliumToAffineLowerPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
      registry.insert<mlir::AffineDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final;
};
}

void HeliumToAffineLowerPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<Helium::HeliumDialect>();
  target.addLegalDialect<mlir::AffineDialect, mlir::BuiltinDialect,
    mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  target.addDynamicallyLegalOp<Helium::PrintOp>([](Helium::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(),
                           [](mlir::Type type) { return type.isa<mlir::TensorType>(); });
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ConstantOpLowering, PrintOpLowering>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> Helium::createLowerToAffinePass() {
  return std::make_unique<HeliumToAffineLowerPass>();
}