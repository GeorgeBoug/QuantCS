add_executable(test_bell_ghz test_bell_ghz.cpp)
target_link_libraries(test_bell_ghz PRIVATE quantcs)

add_test(NAME bell_and_ghz_states COMMAND test_bell_ghz)

if(QUANTCS_CUDA_ENABLED)
    add_executable(test_gpu_matches_cpu test_gpu_matches_cpu.cpp)
    target_link_libraries(test_gpu_matches_cpu PRIVATE quantcs)

    add_test(NAME gpu_matches_cpu COMMAND test_gpu_matches_cpu)
endif()

add_executable(test_grover test_grover.cpp)
target_link_libraries(test_grover PRIVATE quantcs)

add_test(NAME grover_search COMMAND test_grover)
