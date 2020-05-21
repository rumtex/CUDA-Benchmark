cmake_minimum_required(VERSION 3.5)

SET(C /usr/bin/cc)
SET(CMAKE_C_COMPILER /usr/bin/cc)
SET(CXX /usr/bin/clang++-9)
SET(CMAKE_CXX_COMPILER /usr/bin/clang++-9)

# Set the project name
set(PROJECT pascal-example)
project(${PROJECT})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

option(DEBUG_LOG   "Debug logs"                ON)
option(BUILD_TESTS "Build libs test binary"    ON)

###
# sources
###
file(GLOB_RECURSE src src/*.cpp)

# binary
add_executable(${PROJECT}
    ${src}
)

# deps
target_link_libraries(${PROJECT}
    pthread
)

# -I/directory/path/
target_include_directories(${PROJECT}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

if (DEBUG_LOG)
    ADD_DEFINITIONS(-DDEBUG)
endif()

if (BUILD_TESTS)
    set(TEST tests)
    project(${TEST})

    file(GLOB_RECURSE test_src
        tests/*.cpp
        src/libs/*.cpp)

    # binary
    add_executable(${TEST}
        ${test_src}
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