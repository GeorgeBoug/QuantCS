# invmul -- bidirectional array multiplier / factoring by local search.
#
# Self-contained module: it shares the parent project's build (compiler
# settings, OpenMP, CTest registration) but does not link against, include,
# or otherwise depend on the quantum simulator targets. Everything below
# only names files inside invmul/.

add_library(invmul
    src/multiplier.cpp
    src/solver.cpp
)
target_include_directories(invmul PUBLIC include)
target_link_libraries(invmul PUBLIC OpenMP::OpenMP_CXX)
if(MSVC)
    target_compile_options(invmul PUBLIC /EHsc)
endif()

# Demo binary: forward multiply, then clamp the output and factor it back.
add_executable(invmul_demo src/main.cpp)
target_link_libraries(invmul_demo PRIVATE invmul)

add_subdirectory(tests)
add_subdirectory(benchmarks)
