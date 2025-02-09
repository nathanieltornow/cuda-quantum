set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

add_llvm_executable(custom-tool custom-tool.cpp)

llvm_update_compile_flags(custom-tool)
target_link_libraries(custom-tool
  PUBLIC
  MLIRLLVMDialect
  MLIROptLib
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect

  CCDialect
  QuakeDialect
  OptCodeGen
  OptTransforms
  CUDAQSupport
)

export_executable_symbols_for_plugins(custom-tool)
mlir_check_all_link_libraries(custom-tool)

install(TARGETS custom-tool DESTINATION bin)
