cmake_minimum_required(VERSION 3.0)
project(cvortex)
set(CVORTEX_VERSION_MAJOR 0)
set(CVORTEX_VERSION_MINOR 4)
set(CVORTEX_VERSION_PATCH 0)
set(CVORTEX_VERSION ${CVORTEX_VERSION_MAJOR}.${CVORTEX_VERSION_MINOR}.${CVORTEX_VERSION_PATCH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CVORTEX_BUILD_UNIT_TESTS "Builds tests" OFF)
option(CVORTEX_BUILD_BENCHMARKS "Builds benchmarks" OFF)
option(CVORTEX_USE_OPENMP "Use the OpenMP multithreading" ON)
option(CVORTEX_USE_OPENCL "Use OpenCL gpgpu computing" ON)
option(CVORTEX_BUILD_STATIC_LIBRARY "Builds static library instead of shared" OFF)

# Everything is placed in the one dictionary. Life is easier.
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
file (GLOB CVORTEX_INCLUDE "include/cvortex/libcvtx.h")
file (GLOB CVORTEX_SOURCE  "src/*.h" "src/*.hpp" "src/*.cpp")
file (GLOB CVORTEX_OPENCL  "src/*.cl")
source_group("" FILES ${cvortex})
source_group("include" FILES ${CVORTEX_INCLUDE})
source_group("source" FILES ${CVORTEX_SOURCE})
source_group("opencl" FILES ${CVORTEX_OPENCL})

include_directories (include/cvortex)
if(CVORTEX_BUILD_STATIC_LIBRARY)
	add_library(cvortex ${CVORTEX_INCLUDE} ${CVORTEX_SOURCE})
	target_compile_definitions(cvortex PRIVATE CVTX_EXPORT=)
    set_property(TARGET cvortex PROPERTY CXX_STANDARD 17)
else()
	add_library(cvortex SHARED ${CVORTEX_INCLUDE} ${CVORTEX_SOURCE})
	if(WIN32)
		target_compile_definitions(cvortex PRIVATE CVTX_EXPORT=__declspec\(dllexport\))
		target_sources(cvortex PRIVATE tools/natvis/Eigen.natvis)
	else()
		target_compile_definitions(cvortex PRIVATE CVTX_EXPORT=)
	endif()
endif(CVORTEX_BUILD_STATIC_LIBRARY)
# We don't want warnings...
target_compile_definitions(cvortex PRIVATE _CRT_SECURE_NO_WARNINGS)
target_compile_definitions(cvortex PRIVATE  CVORTEX_VERSION=${CVORTEX_VERSION}
											CVORTEX_VERSION_MAJOR=${CVORTEX_VERSION_MAJOR}
											CVORTEX_VERSION_MINOR=${CVORTEX_VERSION_MINOR}
											CVORTEX_VERSION_PATCH=${CVORTEX_VERSION_PATCH})
											
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(cvortex PRIVATE Eigen3::Eigen)

if(CVORTEX_USE_OPENMP)
	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)
		target_link_libraries(cvortex PUBLIC OpenMP::OpenMP_CXX)
		target_compile_definitions(cvortex PRIVATE CVTX_USING_OPENMP)
	endif()
endif(CVORTEX_USE_OPENMP)

if(CVORTEX_USE_OPENCL)
    find_package(OpenCL REQUIRED)
    target_link_libraries(cvortex PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(cvortex PRIVATE ${OpenCL_INCLUDE_DIRS})
	target_compile_definitions(cvortex PRIVATE CVTX_USING_OPENCL)
endif(CVORTEX_USE_OPENCL)

find_package(bsv CONFIG REQUIRED)
target_link_libraries(cvortex PUBLIC bsv)
target_include_directories(cvortex PUBLIC bsv)
#set_target_properties(cvortex PROPERTIES LINKER_LANGUAGE CXX)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    target_link_libraries(cvortex PUBLIC m)   # Maths std library.
endif()
						
set_property(TARGET cvortex PROPERTY FOLDER "libraries")
set_target_properties(cvortex PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
                      PUBLIC_HEADER "${CVORTEX_INCLUDE}")

if(CVORTEX_BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif(CVORTEX_BUILD_UNIT_TESTS)

if(CVORTEX_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif(CVORTEX_BUILD_BENCHMARKS)
