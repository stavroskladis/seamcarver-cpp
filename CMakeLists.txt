cmake_minimum_required(VERSION 3.27)

## vcpkg
include(FetchContent)
include(cmake/vcpkg.cmake)

## project
project (SeamCarver DESCRIPTION "Content-aware image resizing with seam carving" LANGUAGES CXX)

# Require C++17 for std::clamp and other modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for clangd and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to a Debug build if no build type was specified from the command line
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Show coloured compiler diagnostics with file and line information
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # Common flags for all build types
    add_compile_options(
        -Wall            # all warnings
        -Wextra          # extra warnings
        -pedantic        # strict ISO compliance warnings
        -fdiagnostics-color=always      # coloured diagnostics in terminals that support it
    )
    
    # Clang-specific flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-fdiagnostics-absolute-paths)  # print absolute paths to files in diagnostics
    endif()
    
    # Debug-specific flags
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g)  # debug symbols only in Debug builds
    endif()
endif()

## Find dependencies
# libraries list
set(libraries)

find_package(OpenGL REQUIRED)
find_package(imgui CONFIG REQUIRED)
set(libraries ${libraries} imgui::imgui)
find_package(glad CONFIG REQUIRED)
set(libraries ${libraries} glad::glad)
find_package(spdlog CONFIG REQUIRED)
set(libraries ${libraries} spdlog::spdlog)
find_package(fmt CONFIG REQUIRED)
set(libraries ${libraries} fmt::fmt)
find_package(OpenCV CONFIG REQUIRED)
set(libraries ${libraries} opencv_imgproc)

## Create main executable
add_executable(
    SeamCarver
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/glfw_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/image_processing.cpp
)

target_include_directories(
    SeamCarver
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(SeamCarver PRIVATE ${libraries})

# encode asset path
target_compile_definitions(SeamCarver PRIVATE ASSET_PATH="${CMAKE_SOURCE_DIR}/assets")
target_compile_definitions(SeamCarver PRIVATE FMT_HEADER_ONLY)

# Define DEBUG macro for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(SeamCarver PRIVATE DEBUG)
endif()
