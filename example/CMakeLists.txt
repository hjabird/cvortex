cmake_minimum_required(VERSION 3.1)

link_libraries (cvortex m)

add_executable (VortexRings VortexRings.c)
add_executable (VortexTunnel vortex_tunnel.c)
