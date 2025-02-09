#include <queue>

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;



namespace {

class CircuitSplitterPass
    : public PassWrapper<CircuitSplitterPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CircuitSplitterPass)

  llvm::StringRef getArgument() const override { return "split-circuits"; }

private:
  // Track classical values used by quantum operations
  llvm::DenseSet<mlir::Value> getClassicalDependencies(Operation *op) {
    llvm::DenseSet<mlir::Value> dependencies;

    // Add all operands that aren't qubits
    for (auto operand : op->getOperands()) {
      if (!operand.getType().isa<quake::RefType>() &&
          !operand.getType().isa<quake::VeqType>()) {
        dependencies.insert(operand);
      }
    }
    return dependencies;
  }

  // Check if operation uses any of the qubits in the group
  bool shouldClone(Operation *op, const llvm::DenseSet<mlir::Value> &qubits) {
    // Never clone function operations
    if (isa<func::FuncOp>(op)) {
      return false;
    }

    // If it's not a quantum operation or measure, include it
    if (!op->hasTrait<cudaq::QuantumGate>()) { // &&
      // !isa<quake::MeasureOp>(op)) {
      return true;
    }

    if (auto extractOp = dyn_cast<quake::ExtractRefOp>(op)) {
      if (qubits.count(extractOp)) {
        return true;
      }
      return false;
    }

    if (op->hasTrait<cudaq::QuantumMeasure>()) {
      return false;
    }

    // For quantum operations, check if they use qubits from the group
    if (auto opInterface = dyn_cast<quake::OperatorInterface>(op)) {
      auto controls = opInterface.getControls();
      auto targets = opInterface.getTargets();

      // Check if operation uses any qubit from our group
      for (auto qubit : controls)
        if (qubits.count(qubit))
          return true;

      for (auto qubit : targets)
        if (qubits.count(qubit))
          return true;

      // If quantum operation but uses no qubits from our group, exclude it
      return false;
    }

    return true; // Include any other operations by default
  }

  // Create a new function for the given qubit group
  func::FuncOp createGroupFunction(OpBuilder &builder,
                                   func::FuncOp originalFunc,
                                   const llvm::DenseSet<mlir::Value> &qubits,
                                   size_t groupId) {
    // Create function name
    std::string newFuncName = originalFunc.getName().str() +
                              "_circuit_tensor_" + std::to_string(groupId);

    // Create function type with same signature as original
    auto funcType = originalFunc.getFunctionType();

    // Create the new function
    auto newFunc =
        func::FuncOp::create(originalFunc.getLoc(), newFuncName, funcType);

    // Create entry block
    Block *entryBlock = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Initialize valueMap with function arguments
    llvm::DenseMap<Value, Value> valueMap;
    for (auto [oldArg, newArg] :
         llvm::zip(originalFunc.getArguments(), entryBlock->getArguments())) {
      valueMap[oldArg] = newArg;
    }

    // Track which operations we need to clone
    llvm::DenseSet<Operation *> opsToClone;
    llvm::DenseSet<mlir::Value> requiredValues;

    // First pass - identify quantum operations and their classical dependencies
    originalFunc.walk([&](Operation *op) {
      if (shouldClone(op, qubits)) {
        opsToClone.insert(op);
        // Add classical dependencies
        auto deps = getClassicalDependencies(op);
        requiredValues.insert(deps.begin(), deps.end());
      }
    });

    // Second pass - add operations that produce required classical values
    originalFunc.walk([&](Operation *op) {
      for (auto result : op->getResults()) {
        if (requiredValues.count(result)) {
          opsToClone.insert(op);
        }
      }
    });

    // Clone operations in order, using valueMap for operands
    builder.setInsertionPointToStart(entryBlock);
    mlir::IRMapping mapping; // Create mapping outside the loop

    // Map the block arguments first
    for (auto [oldArg, newArg] :
         llvm::zip(originalFunc.getArguments(), entryBlock->getArguments())) {
      mapping.map(oldArg, newArg);
    }

    originalFunc.walk([&](Operation *op) {
      if (opsToClone.count(op)) {
        // Clone operation with the mapping
        auto *newOp = builder.clone(*op, mapping);

        // Map results for future uses
        for (auto [oldRes, newRes] :
             llvm::zip(op->getResults(), newOp->getResults())) {
          mapping.map(oldRes, newRes);
        }
      }
    });

    return newFunc;
  }

public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> qubitGraph;

    funcOp.walk([&](Operation *op) {
      if (!op->hasTrait<cudaq::QuantumGate>()) {
        return;
      }

      // Get operator interface which all quantum gates implement
      auto opInterface = dyn_cast<quake::OperatorInterface>(op);

      // Get control and target qubits
      auto controls = opInterface.getControls();
      auto targets = opInterface.getTargets();

      // chain control and target qubits
      SmallVector<Value> qubits;
      qubits.append(controls.begin(), controls.end());
      qubits.append(targets.begin(), targets.end());

      if (qubits.size() == 1) {
        qubitGraph[qubits[0]] = {};
      }

      else if (qubits.size() == 2) {
        qubitGraph[qubits[0]].push_back(qubits[1]);
        qubitGraph[qubits[1]].push_back(qubits[0]);
      }

      else {
        llvm::errs() << "Unsupported gate with more than 2 qubits\n";
      }
    });

    llvm::DenseMap<uint16_t, llvm::DenseSet<mlir::Value>> qubitGroups;
    llvm::DenseSet<mlir::Value> visited;

    size_t group = 0;
    for (auto &[qubit, _] : qubitGraph) {
      if (visited.count(qubit)) {
        continue;
      }

      // bfs traversal
      std::queue<mlir::Value> q;
      q.push(qubit);

      while (!q.empty()) {
        auto current = q.front();
        q.pop();

        visited.insert(current);
        qubitGroups[group].insert(current);

        for (auto neighbor : qubitGraph[current]) {
          if (!visited.count(neighbor)) {
            q.push(neighbor);
          }
        }
      }
      group++;
    }

    if (group <= 1) {
      return;
    }

    // After building qubitGroups, create new functions
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    builder.setInsertionPoint(funcOp);

    // Create new function for each group
    for (auto &[groupId, qubits] : qubitGroups) {
      auto newFunc = createGroupFunction(builder, funcOp, qubits, groupId);
      moduleOp.push_back(newFunc);
    }

    // Insert calls to split functions at the beginning
    Block &entryBlock = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);

    // Create calls to all split functions in order
    for (size_t i = 0; i < group; i++) {
      std::string funcName =
          funcOp.getName().str() + "_circuit_tensor_" + std::to_string(i);
      auto calleeOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
      if (!calleeOp) {
        llvm::errs() << "Failed to find split function: " << funcName << "\n";
        continue;
      }

      // Create call with all original arguments
      builder.create<func::CallOp>(funcOp.getLoc(), calleeOp,
                                   entryBlock.getArguments());
    }
  }
};

// Declare pass creation function
// std::unique_ptr<Pass> createCircuitSplitterPass() {
//   return std::make_unique<CircuitSplitterPass>();
// }

} // namespace

CUDAQ_REGISTER_MLIR_PLUGIN(CircuitSplitter, []() {
  mlir::PassRegistration<CircuitSplitterPass>();
})
