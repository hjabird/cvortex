cmake_minimum_required(VERSION 3.0)
project(cvortex)

option(BUILD_UNIT_TESTS "Builds tests" OFF)
option(BUILD_EXAMPLES "Builds examples" OFF)
option(USE_OPENMP "Use the OpenMP multithreading" ON)
option(USE_OPENCL "Use OpenCL gpgpu computing" OFF)
option(BUILD_STATIC_LIBRARY "Builds static library instead of shared" OFF)

# Everything is placed in the one dictionary. Life is easier.
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif(USE_OPENMP)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
file (GLOB CVORTEX_INCLUDE "include/cvortex/*.h") # So shoot me for GLOBing.
file (GLOB CVORTEX_SOURCE  "src/*.c")
source_group("" FILES ${cvortex})
source_group("include" FILES ${CVORTEX_INCLUDE})
source_group("source" FILES ${CVORTEX_SOURCE})

include_directories (include/cvortex)
if(BUILD_STATIC_LIBRARY)
	add_library(cvortex ${CVORTEX_INCLUDE} ${CVORTEX_SOURCE})
	target_compile_definitions(cvortex PRIVATE CVTX_EXPORT)
else()
	add_library(cvortex SHARED ${CVORTEX_INCLUDE} ${CVORTEX_SOURCE})
	if(WIN32)
		target_compile_definitions(cvortex PRIVATE CVTX_EXPORT=__declspec\(dllexport\))
	else()
		target_compile_definitions(cvortex PRIVATE CVTX_EXPORT)
	endif()
endif(BUILD_STATIC_LIBRARY)
# We don't want warnings...
target_compile_definitions(cvortex PRIVATE _CRT_SECURE_NO_WARNINGS)

if(USE_OPENCL)
    find_package(OpenCL REQUIRED)
    target_link_libraries(cvortex PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(cvortex PRIVATE ${OpenCL_INCLUDE_DIRS})
	target_compile_definitions(cvortex PRIVATE CVTX_USING_OPENCL)
endif(USE_OPENCL)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    link_libraries(cvortex m)   # Maths std library.
endif()
						
set_property(TARGET cvortex PROPERTY FOLDER "libraries")
set_target_properties(cvortex PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
                      PUBLIC_HEADER "${CVORTEX_INCLUDE}")

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif(BUILD_UNIT_TESTS)

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif(BUILD_EXAMPLES)

