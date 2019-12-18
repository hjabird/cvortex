cmake_minimum_required(VERSION 3.1)

# Find all of our benchmark sources
file (GLOB BENCHCVTX_SOURCE  "*.[ch]")
source_group("cvtx_benchmark_source" FILES ${BENCHCVTX_SOURCE})
add_executable(all_bench ${BENCHCVTX_SOURCE})

target_link_libraries(all_bench PUBLIC cvortex)
target_link_libraries(all_bench PUBLIC bsv)

enable_testing()
target_compile_definitions(all_bench PRIVATE _CRT_SECURE_NO_WARNINGS)


