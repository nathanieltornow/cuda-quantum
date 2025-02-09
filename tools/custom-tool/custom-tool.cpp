#include "cudaq/Optimizer/CodeGen/CodeGenDialect.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Common/InlinerInterface.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Plugin.h"
#include "cudaq/Support/Version.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"

using namespace llvm;

/// @brief Add a command line flag for loading plugins
static cl::list<std::string>
    CudaQPlugins("load-cudaq-plugin",
                 cl::desc("Load CUDA-Q plugin by specifying its library"));

int main(int argc, char **argv) {
  // Set the bug report message
  llvm::setBugReportMsg(cudaq::bugReportMsg);

  // Initialize LLVM
  llvm::InitLLVM initLLVM(argc, argv);

  // Register all CUDA Quantum passes
  cudaq::registerAllPasses();

  // Debug: Print available passes
  llvm::outs() << "Available passes:\n";

  llvm::outs() << "\n";

  // Parse command line options
  cl::ParseCommandLineOptions(argc, argv, "CUDA Quantum Custom Tool\n");

  // Load plugins if specified
  for (const auto &pluginPath : CudaQPlugins) {
    auto Plugin = cudaq::Plugin::Load(pluginPath);
    if (!Plugin) {
      llvm::errs() << "Failed to load plugin: " << pluginPath << "\n";
      return 1;
    }
    Plugin.get().registerExtensions();
  }

  // Set up dialect registry and context
  mlir::DialectRegistry registry;
  cudaq::registerAllDialects(registry);
  registry.insert<cudaq::codegen::CodeGenDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.printOpOnDiagnostic(true); // Enable operation printing in diagnostics

  // Parse input
  auto bufferOrErr = llvm::MemoryBuffer::getSTDIN();
  if (auto err = bufferOrErr.getError()) {
    llvm::errs() << "Failed to read from stdin: " << err.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(bufferOrErr.get()), llvm::SMLoc());
  auto moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!moduleOp) {
    llvm::errs() << "Failed to parse input module\n";
    return 1;
  }

  // Create pass manager with debug enabled
  mlir::PassManager pm(&context);
  pm.enableVerifier(); // Verify IR after each pass

  // Add pass pipeline with debug output
  if (mlir::failed(
          mlir::parsePassPipeline("cse,func.func(mid-measure-analysis)", pm))) {
    llvm::errs() << "Failed to parse pass pipeline\n";
    return 1;
  }

  // Run the passes
  if (mlir::failed(pm.run(*moduleOp))) {
    llvm::errs() << "Failed to run passes\n";
    return 1;
  }

  // Print the final module
  moduleOp->print(llvm::outs());
  return 0;
}
