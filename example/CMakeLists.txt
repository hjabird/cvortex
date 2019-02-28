cmake_minimum_required(VERSION 3.1)

find_package(bsv CONFIG REQUIRED)
link_libraries(cvortex bsv)
if (${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
    link_libraries(m)   # Maths std library.
endif()

add_executable (VortexRings VortexRings.c)
target_compile_definitions(VortexRings PRIVATE _CRT_SECURE_NO_WARNINGS)
add_executable (VortexTunnel vortex_tunnel.c)
target_compile_definitions(VortexTunnel PRIVATE _CRT_SECURE_NO_WARNINGS)
add_executable (BenchGaussian bench_gaussian.c)
target_compile_definitions(BenchGaussian PRIVATE _CRT_SECURE_NO_WARNINGS)
