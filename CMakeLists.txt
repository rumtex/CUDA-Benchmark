cmake_minimum_required(VERSION 3.5)

option(DEBUG_LOG   "Debug logs"                ON)
option(BUILD_TESTS "Build libs test binary"    OFF) #сломаны из-за CUDA

SET(CMAKE_C_COMPILER /usr/bin/gcc-8)
SET(CMAKE_CXX_COMPILER /usr/bin/g++-8)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Set the project name
set(PROJECT pascal-example)
project(${PROJECT} C CXX CUDA)

# target_compile_features(${PROJECT} PUBLIC cxx_std_14)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin g++")

###
# sources
###
file(GLOB_RECURSE SRC_CUDA src/*.cu)
file(GLOB_RECURSE SRC_CXX src/*.cpp)
file(GLOB_RECURSE SRC_C src/*.c)

# binary
add_executable(${PROJECT}
    ${SRC_C}
    ${SRC_CUDA}
    ${SRC_CXX}
)

# deps
target_link_libraries(${PROJECT}
    pthread
    m
    stdc++
    cudart
)

# -I/directory/path/
target_include_directories(${PROJECT}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/cuda
)

# We need to explicitly state that we need all CUDA files to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(${PROJECT}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

if (DEBUG_LOG)
    ADD_DEFINITIONS(-DDEBUG)
endif()

if (BUILD_TESTS)
    set(TEST tests)
    project(${TEST} C CXX)

    file(GLOB_RECURSE test_src
        tests/*.cpp
        src/libs/*.c
        src/libs/*.cpp
        )

    # binary
    add_executable(${TEST}
        ${test_src}
    )

    # deps
    target_link_libraries(${TEST}
        m
        stdc++
    )

    target_include_directories(${TEST}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/tests
    )
    target_include_directories(${TEST}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
    )
endif()