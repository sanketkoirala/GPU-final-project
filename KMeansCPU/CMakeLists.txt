set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# CPU test executable
add_executable(kmeans_cpu_test
    "${SRC_PATH}/main_test.cpp"
    "${SRC_PATH}/kmeans_cpu.cpp")
target_include_directories(kmeans_cpu_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(kmeans_cpu_test PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(kmeans_cpu_test ${COMMON_LIBS})
add_dependencies(kmeans_cpu_test libgputk)

# Datagen executable
add_executable(kmeans_datagen
    "${SRC_PATH}/dataset_generator.cpp"
    "${SRC_PATH}/kmeans_cpu.cpp")
target_include_directories(kmeans_datagen PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(kmeans_datagen PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(kmeans_datagen ${COMMON_LIBS})
add_dependencies(kmeans_datagen libgputk)

# Set output directories
set_target_properties(kmeans_cpu_test kmeans_datagen
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")
