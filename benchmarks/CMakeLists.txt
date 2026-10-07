add_executable(bench_gates bench_gates.cpp)
target_link_libraries(bench_gates PRIVATE quantcs)

add_executable(bench_scaling bench_scaling.cpp)
target_link_libraries(bench_scaling PRIVATE quantcs)
