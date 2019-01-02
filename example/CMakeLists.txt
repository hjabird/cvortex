cmake_minimum_required(VERSION 3.1)

link_libraries (cvortex)

add_executable (VortexRings VortexRings.c)
target_compile_definitions(VortexRings PRIVATE _CRT_SECURE_NO_WARNINGS)
add_executable (VortexTunnel vortex_tunnel.c)
target_compile_definitions(VortexTunnel PRIVATE _CRT_SECURE_NO_WARNINGS)
add_executable (BenchGaussian bench_gaussian.c)
target_compile_definitions(BenchGaussian PRIVATE _CRT_SECURE_NO_WARNINGS)
