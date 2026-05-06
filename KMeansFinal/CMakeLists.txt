set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Collect source files
file(GLOB CUDA_SOURCES "${SRC_PATH}/*.cu")

# Create main executable
add_executable(kmeans_final_test ${CUDA_SOURCES} "${SRC_PATH}/main_test.cpp")
target_include_directories(kmeans_final_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(kmeans_final_test PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(kmeans_final_test ${COMMON_LIBS})
add_dependencies(kmeans_final_test libgputk)

# Set output directories
set_target_properties(kmeans_final_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")
