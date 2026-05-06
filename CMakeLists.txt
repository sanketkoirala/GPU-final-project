# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(FetchContent)
include(ExternalProject)

# Allow the user to specify a desired compiler version and install path
set(USER_SPECIFIED_C_COMPILER "" CACHE STRING "User-specified C compiler")
set(USER_SPECIFIED_CXX_COMPILER "" CACHE STRING "User-specified C++ compiler")
set(GPUTK_INSTALL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libgputk" CACHE PATH "Installation path for libgputk")

# Set compilers before project declaration
if(NOT "${USER_SPECIFIED_CXX_COMPILER}" STREQUAL "")
    set(CMAKE_CXX_COMPILER "${USER_SPECIFIED_CXX_COMPILER}")
endif()

if(NOT "${USER_SPECIFIED_C_COMPILER}" STREQUAL "")
    set(CMAKE_C_COMPILER "${USER_SPECIFIED_C_COMPILER}")
endif()

# Set CUDA flags before project declaration
set(CMAKE_CUDA_COMPILER_FLAGS "-allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
if(NOT "${USER_SPECIFIED_C_COMPILER}" STREQUAL "")
    set(CMAKE_CUDA_HOST_COMPILER "${USER_SPECIFIED_C_COMPILER}")
endif()

project(cuda_projects LANGUAGES CXX CUDA)

# Set C++ standard for all targets
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Function to find libgputk
function(find_libgputk)
    # First check user-specified path
    if(NOT "${SYSTEM_GPUTK_PATH}" STREQUAL "")
        if(EXISTS "${SYSTEM_GPUTK_PATH}/include/gputk.h")
            set(GPUTK_FOUND TRUE PARENT_SCOPE)
            set(GPUTK_INCLUDE_DIR "${SYSTEM_GPUTK_PATH}/include" PARENT_SCOPE)
            set(GPUTK_LIBRARY_DIR "${SYSTEM_GPUTK_PATH}/lib" PARENT_SCOPE)
            message(STATUS "Found libgputk in specified path: ${SYSTEM_GPUTK_PATH}")
            return()
        endif()
    endif()

    # Check common system paths
    find_path(GPUTK_INCLUDE_DIR gputk.h
        PATHS
        /usr/include
        /usr/local/include
        /opt/local/include
	    /usr/local/libgputk/include
    )

    find_library(GPUTK_LIBRARY
        NAMES
        gputk
        libgputk
        PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
	    /usr/local/libgputk/lib
	    /usr/local/libgputk/
    )

    if(GPUTK_INCLUDE_DIR AND GPUTK_LIBRARY)
        get_filename_component(GPUTK_LIBRARY_DIR ${GPUTK_LIBRARY} DIRECTORY)
        set(GPUTK_FOUND TRUE PARENT_SCOPE)
        set(GPUTK_INCLUDE_DIR ${GPUTK_INCLUDE_DIR} PARENT_SCOPE)
        set(GPUTK_LIBRARY_DIR ${GPUTK_LIBRARY_DIR} PARENT_SCOPE)
        set(GPUTK_LIBRARY ${GPUTK_LIBRARY} PARENT_SCOPE)
	add_library(libgputk SHARED IMPORTED GLOBAL)
	set_target_properties(libgputk PROPERTIES
        IMPORTED_LOCATION ${GPUTK_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${GPUTK_INCLUDE_DIR}
    )
        message(STATUS "Found system libgputk:")
        message(STATUS "    include: ${GPUTK_INCLUDE_DIR}")
        message(STATUS "    library: ${GPUTK_LIBRARY_DIR}")
    else()
        set(GPUTK_FOUND FALSE PARENT_SCOPE)
        message(STATUS "System libgputk not found, will build from source")
    endif()
endfunction()

# Try to find libgputk
find_libgputk()

message(STATUS "GPUTK_FOUND: ${GPUTK_INSTALL_PATH}")

# Build libgputk if not found
if(NOT GPUTK_FOUND)
    ExternalProject_Add(libgputk
        GIT_REPOSITORY https://github.com/ajdillhoff/libgputk.git
        GIT_TAG main
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${GPUTK_INSTALL_PATH}
            -DBUILD_SHARED_LIBS=ON
        BUILD_ALWAYS OFF
        TEST_COMMAND ""
    )
    set(GPUTK_INCLUDE_DIR "${GPUTK_INSTALL_PATH}/include")
    set(GPUTK_LIBRARY_DIR "${GPUTK_INSTALL_PATH}/lib")
endif()

# Set common variables for all subprojects
set(COMMON_INCLUDE_DIRS
    ${GPUTK_INSTALL_PATH}
    ${GPUTK_INSTALL_PATH}/libgputk/include
    ${GPUTK_INCLUDE_DIR}
)

set(COMMON_LINK_DIRS
    ${GPUTK_INSTALL_PATH}/libgputk/lib
)

set(COMMON_LIBS
    libgputk
    m
    stdc++
    CUDA::cudart
)

# Additional CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -allow-unsupported-compiler
)

# Add subdirectories - all phases implemented
add_subdirectory(KMeansCPU)
add_subdirectory(KMeansBasic)
add_subdirectory(KMeansCMem)
add_subdirectory(KMeansShared)
add_subdirectory(KMeansCoarse)
add_subdirectory(KMeansCoalesced)
add_subdirectory(KMeansFinal)
