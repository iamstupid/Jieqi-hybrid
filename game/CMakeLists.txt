cmake_minimum_required(VERSION 3.15...3.27)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# =============================================================================
# BUILD CONFIGURATION
# =============================================================================

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui/ccmake
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Print build type for clarity
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# COMPILER FLAGS CONFIGURATION
# =============================================================================

# Base optimization flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# Architecture-specific optimizations (applied to all builds)
set(ARCH_FLAGS
        -mbmi2
        -mbmi
        -mpopcnt
        -mlzcnt
        -msse4.2
        -mavx2
)

# Additional debug flags
set(DEBUG_FLAGS
        -fno-omit-frame-pointer
        -fno-inline-functions-called-once
)

# Additional release flags
set(RELEASE_FLAGS
        -fomit-frame-pointer
        -finline-functions
        -flto  # Link Time Optimization
)

# =============================================================================
# FIND PACKAGES
# =============================================================================

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# =============================================================================
# SOURCE FILES
# =============================================================================

file(GLOB_RECURSE CHESS_SOURCES "src/chess/*.cpp" "src/chess/*.cxx" "src/chess/*.cc")
file(GLOB_RECURSE UTILS_SOURCES "src/utils/*.cpp" "src/utils/*.cxx" "src/utils/*.cc")
file(GLOB_RECURSE NN_SOURCES "src/nn/*.cpp" "src/nn/*.cc")
file(GLOB_RECURSE SEARCH_SOURCES "src/search/*.cpp" "src/search/*.cc")

# =============================================================================
# CORE LOGIC LIBRARY
# =============================================================================

add_library(jieqi_game_core STATIC
        ${CHESS_SOURCES}
        ${UTILS_SOURCES}
        ${NN_SOURCES}
        ${SEARCH_SOURCES}
        src/version.cc
)

target_compile_features(jieqi_game_core PUBLIC cxx_std_20)
set_target_properties(jieqi_game_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Apply architecture flags to all builds
target_compile_options(jieqi_game_core PUBLIC ${ARCH_FLAGS})

# Apply build-type specific flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(jieqi_game_core PUBLIC ${DEBUG_FLAGS})
    message(STATUS "Debug build: Adding debug flags")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(jieqi_game_core PUBLIC ${RELEASE_FLAGS})
    message(STATUS "Release build: Adding optimization flags")
endif()

target_include_directories(jieqi_game_core PUBLIC
        src
        src/chess
        src/utils
        src/nn
        src/search
)

target_compile_definitions(jieqi_game_core PUBLIC VERSION_INFO=${PROJECT_VERSION})

# =============================================================================
# PYTHON MODULE
# =============================================================================

python_add_library(jieqi_game MODULE src/main.cpp WITH_SOABI)
target_link_libraries(jieqi_game PRIVATE jieqi_game_core pybind11::headers)

# Apply the same build-type specific settings to the Python module
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(jieqi_game PRIVATE ${DEBUG_FLAGS})
    set_target_properties(jieqi_game PROPERTIES
            COMPILE_FLAGS "-g -O0"
            LINK_FLAGS "-g"
    )
    message(STATUS "Python module: Debug configuration")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(jieqi_game PRIVATE ${RELEASE_FLAGS})
    message(STATUS "Python module: Release configuration")
endif()

# =============================================================================
# TEST EXECUTABLE
# =============================================================================

add_executable(run_tests test/run_tests.cpp)
target_link_libraries(run_tests PRIVATE jieqi_game_core)

# Apply build-type specific flags to test executable
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(run_tests PRIVATE ${DEBUG_FLAGS})
endif()

# =============================================================================
# INSTALLATION
# =============================================================================

install(TARGETS jieqi_game DESTINATION .)