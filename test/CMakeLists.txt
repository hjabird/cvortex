cmake_minimum_required(VERSION 3.1)

# Find all of our tests sources
file (GLOB TESTCVTX_SOURCE  "*.[ch]")
source_group("cvtx_test_source" FILES ${TESTCVTX_SOURCE})
add_executable(all_tests ${TESTCVTX_SOURCE})

target_link_libraries(all_tests PUBLIC cvortex)
target_link_libraries(all_tests PUBLIC bsv)

enable_testing()
target_compile_definitions(all_tests PRIVATE _CRT_SECURE_NO_WARNINGS)


