#include "cudaq/Optimizer/Dialect/Common/Traits.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

class MidMeasureAnalysis
    : public PassWrapper<MidMeasureAnalysis, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MidMeasureAnalysis)

  llvm::StringRef getArgument() const override {
    return "mid-measure-analysis";
  }

  llvm::StringRef getDescription() const override {
    return "Check for operations after measurements on the same qubit";
  }

  std::pair<Value, int64_t> getQubit(Operation *op, int64_t index) {
    Value allocOperand = nullptr;
    int64_t qubitIndex = -1;

    // Validate the operand at the specified index
    Value operand = op->getOperand(index);
    if (!operand) {
      llvm::errs() << "Operand not found\n";
      return {nullptr, -1};
    }

    // Get the defining operation of the operand
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      llvm::errs() << "Defining operation not found\n";
      return {nullptr, -1};
    }

    if (auto extractOp = dyn_cast<quake::ExtractRefOp>(defOp)) {
      allocOperand = extractOp.getOperand(0);

      // Determine the index
      if (auto constAttr = extractOp->getAttrOfType<IntegerAttr>("rawIndex")) {
        qubitIndex = constAttr.getValue().getSExtValue();
        if (qubitIndex < 0) {
          llvm::errs() << "WARNING: Invalid qubit index\n";
          return {nullptr, -1};
        }
      }
    } else if (auto allocOp = dyn_cast<quake::AllocaOp>(defOp)) {
      allocOperand = allocOp;
      qubitIndex = -1; // Default index for allocation
    } else {
      llvm::errs() << "Unsupported operation type\n";
    }

    return {allocOperand, qubitIndex};
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Map to store the qubit identifiers for each operation

    llvm::DenseSet<mlir::Value> measurementsResult;
    llvm::DenseMap<mlir::Value, llvm::DenseSet<int64_t>> measuredQubits;

    // std::unordered_multimap<Value, int64_t> measured_qubits();

    bool midMeasure = false;

    funcOp.walk([&](Operation *op) {
      if (!(op->getDialect() && op->getDialect()->getNamespace() == "quake")) {
        return;
      }

      // TODO check whethe any operand is a measurement result
      for (auto op : op->getOperands()) {
        if (measurementsResult.count(op)) {
          midMeasure = true;
          return;
        }
      }

      if (!op->hasTrait<cudaq::QuantumGate>() &&
          !op->hasTrait<cudaq::QuantumMeasure>()) {
        return;
      }

      bool isMeasure = op->hasTrait<cudaq::QuantumMeasure>();

      if (isMeasure) {
        measurementsResult.insert(op->getResult(0));
      }

      // print the operation name
      llvm::outs() << "Operation: " << op->getName() << "\n";

      for (int64_t i = 0; i < op->getNumOperands(); i++) {
        auto [veq, index] = getQubit(op, i);

        llvm::outs() << "  Operand: " << i << " - " << veq << " - " << index
                     << " - " << isMeasure << "\n";

        if (!veq)
          continue;

        auto it = measuredQubits.find(veq);

        if (it != measuredQubits.end()) {
          // we already have a measurement on this qubit
          auto &indices = it->second;

          llvm::outs() << "  Found measured qubits " << veq << " - " << index
                       << " - " << indices.count(index) << "\n";
          if (index <= -1 || indices.count(index)) {
            midMeasure = true;
            return;
          }
        }

        if (isMeasure) {
          llvm::outs() << "  Adding to measured qubits " << veq << " - "
                       << index << "\n";
          if (it == measuredQubits.end()) {
            measuredQubits.insert({veq, {index}});
          } else {
            it->second.insert(index);
          }
        }
      }
    });
    // set attribute to the function
    funcOp->setAttr("mid-measure",
                    BoolAttr::get(funcOp.getContext(), midMeasure));
  }
};

} // namespace
// Register all passes in a single plugin
CUDAQ_REGISTER_MLIR_PLUGIN(MidMeasurePlugin, []() {
  mlir::PassRegistration<MidMeasureAnalysis>();
})
// CUDAQ_REGISTER_MLIR_PASS(MidMeasureAnalysis)
