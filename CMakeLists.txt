cmake_minimum_required(VERSION 3.0)
project(cvortex)

file (GLOB CVORTEX_INCLUDE "include/cvortex/*.h")
file (GLOB CVORTEX_SOURCE  "src/*.c")
source_group("" FILES ${cvortex})
source_group("include" FILES ${CVORTEX_INCLUDE})
source_group("source" FILES ${CVORTEX_SOURCE})

include_directories (include/cvortex)
add_library(cvortex  	${CVORTEX_INCLUDE} 
						${CVORTEX_SOURCE})
						
set_property(TARGET cvortex PROPERTY FOLDER "libraries")
set_target_properties(cvortex PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
                      PUBLIC_HEADER "${CVORTEX_INCLUDE}")
