# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find necessary packages
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Collect all source files for the core logic
file(GLOB_RECURSE CHESS_SOURCES "src/chess/*.cpp" "src/chess/*.cxx" "src/chess/*.cc")
file(GLOB_RECURSE UTILS_SOURCES "src/utils/*.cpp" "src/utils/*.cxx" "src/utils/*.cc")
file(GLOB_RECURSE NN_SOURCES "src/nn/*.cpp" "src/nn/*.cc")
file(GLOB_RECURSE SEARCH_SOURCES "src/search/*.cpp" "src/search/*.cc")

# --- 1. Core Logic Library (STATIC) ---
# Create a STATIC library containing all your game logic.
# This target can be linked into other targets.
add_library(jieqi_game_core STATIC
        ${CHESS_SOURCES}
        ${UTILS_SOURCES}
        ${NN_SOURCES}
        ${SEARCH_SOURCES}
        src/version.cc
)

# Set properties on the core library. Use PUBLIC so they are inherited by linkers.
target_compile_features(jieqi_game_core PUBLIC cxx_std_20)
set_target_properties(jieqi_game_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(jieqi_game_core PUBLIC
        -mbmi2
        -mbmi
        -mpopcnt
        -mlzcnt
        -msse4.2
        -mavx2
)

target_include_directories(jieqi_game_core PUBLIC
        src
        src/chess
        src/utils
        src/nn
        src/search)

target_compile_definitions(jieqi_game_core PUBLIC VERSION_INFO=${PROJECT_VERSION})

# --- 2. Python Module (MODULE) ---
# This target is now just a thin wrapper for the pybind11 bindings.
python_add_library(jieqi_game MODULE
        src/main.cpp
        WITH_SOABI)

# Link the Python module against the core logic and pybind11.
target_link_libraries(jieqi_game PRIVATE jieqi_game_core pybind11::headers)


# --- 3. Test Executable ---
# This target is for your C++ debugging entry point.
add_executable(run_tests test/run_tests.cpp)

# Link the test program against the core logic library. This is now allowed.
target_link_libraries(run_tests PRIVATE jieqi_game_core)


# The install command now only targets the Python module
install(TARGETS jieqi_game DESTINATION .)