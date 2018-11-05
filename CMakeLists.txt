cmake_minimum_required(VERSION 3.0)
project(cvortex)

option(BUILD_UNIT_TESTS "Builds tests" OFF)
option(BUILD_EXAMPLES "Builds examples" OFF)
option(USE_OPENMP "Use the OpenMP multithreading" ON)

if(USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif(USE_OPENMP)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
file (GLOB CVORTEX_INCLUDE "include/cvortex/*.h")
file (GLOB CVORTEX_SOURCE  "src/*.c")
source_group("" FILES ${cvortex})
source_group("include" FILES ${CVORTEX_INCLUDE})
source_group("source" FILES ${CVORTEX_SOURCE})

include_directories (include/cvortex)
add_library(cvortex  	${CVORTEX_INCLUDE} ${CVORTEX_SOURCE})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
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

