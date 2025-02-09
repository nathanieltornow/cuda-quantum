#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "llvm/ADT/Optional.h" // Add this header
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h" // Add this header
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

// Here is an example MLIR Pass that one can write externally and
// use via the cudaq-opt tool, with the --load-cudaq-plugin flag.
// The pass here is simple, replace Hadamard operations with S operations.

using namespace mlir;

namespace {

// Helper function to parse comma-separated list
static std::vector<int> parseIntList(const std::string &str) {
  std::vector<int> result;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    result.push_back(std::stoi(item));
  }
  return result;
}

class CutPlacementPass
    : public PassWrapper<CutPlacementPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CutPlacementPass)

  Option<std::string> cutRange{
      *this, "cut-between",
      llvm::cl::desc(
          "Optional: Specify two qubits to cut gates between (format: q1,q2)")};

  // Store parsed state
  bool hasCutBetween = false;
  size_t qubit1, qubit2;

  llvm::StringRef getArgument() const override { return "cut"; }
  llvm::StringRef getDescription() const override {
    return "Place cuts in quantum circuits based on specified strategy";
  }

  // Constructor
  CutPlacementPass() = default;

  // Copy constructor required by PassWrapper
  CutPlacementPass(const CutPlacementPass &pass)
      : PassWrapper<CutPlacementPass, OperationPass<func::FuncOp>>(pass),
        hasCutBetween(pass.hasCutBetween), qubit1(pass.qubit1),
        qubit2(pass.qubit2) {}

  LogicalResult initialize(MLIRContext *context) override {
    if (!cutRange.hasValue())
      return success(); // Optional parameter not provided

    auto range = parseIntList(cutRange.getValue());
    if (range.size() != 2) {
      return failure();
    }

    hasCutBetween = true;
    qubit1 = range[0];
    qubit2 = range[1];
    return success();
  }

  void runOnOperation() override {
    if (!hasCutBetween) {
      llvm::outs() << "Processing without cut between\n";
      return;
    }

    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    OpBuilder builder(ctx);

    funcOp.walk([&](Operation *op) {
      if (!op->hasTrait<cudaq::QuantumGate>()) {
        return;
      }

      auto opInterface = dyn_cast<quake::OperatorInterface>(op);
      auto controls = opInterface.getControls();
      auto targets = opInterface.getTargets();

      if (controls.size() != 1 || targets.size() != 1) {
        return;
      }

      auto getQubitIndex = [](Value v) -> std::optional<size_t> {
        if (auto extractOp =
                dyn_cast_or_null<quake::ExtractRefOp>(v.getDefiningOp())) {
          if (extractOp.hasConstantIndex())
            return extractOp.getConstantIndex();
        }
        return std::nullopt;
      };

      auto controlIdx = getQubitIndex(controls[0]);
      auto targetIdx = getQubitIndex(targets[0]);

      if (!controlIdx || !targetIdx)
        return;

      bool should_cut = ((*controlIdx == qubit1 && *targetIdx == qubit2) ||
                         (*controlIdx == qubit2 && *targetIdx == qubit1));

      if (should_cut) {
        builder.setInsertionPoint(op);

        // Helper to create a new quantum operation based on type
        auto createNewOp = [&](Value qubit) {
          if (isa<quake::XOp>(op)) {
            // Create empty arrays for parameters and controls
            ValueRange emptyParams;
            ValueRange emptyControls;
            ValueRange targets(qubit);

            // Create dictionary attribute for cuts
            NamedAttribute cutAttr(StringAttr::get(ctx, "cuts"), UnitAttr::get(ctx));
            
            // Build the operation with all required arguments and the cuts attribute
            builder.create<quake::XOp>(op->getLoc(),
                                     /*is_adj=*/false,
                                     /*parameters=*/emptyParams,
                                     /*controls=*/emptyControls,
                                     /*targets=*/targets)
                ->setAttr(cutAttr.getName(), cutAttr.getValue());
          }
        };

        // Create operations for control and target qubits
        createNewOp(controls[0]);
        createNewOp(targets[0]);

        op->erase();
      }
    });
  }
};
} // namespace

// Register the pass and its options
CUDAQ_REGISTER_MLIR_PLUGIN(Cutter,
                           []() { mlir::PassRegistration<CutPlacementPass>(); })
