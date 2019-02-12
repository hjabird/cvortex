cmake_minimum_required(VERSION 3.1)

link_libraries(cvortex)
link_libraries(bsv)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    link_libraries(m)   # Maths std library.
endif()

# Find all of our tests sources
file (GLOB TESTCVTX_SOURCE  "*.c")
source_group("cvtx_test_source" FILES ${TESTCVTX_SOURCE})


# And add the main as an executable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
enable_testing()
add_executable(all_tests ${TESTCVTX_SOURCE})
target_compile_definitions(all_tests PRIVATE _CRT_SECURE_NO_WARNINGS)


