# fetch vcpkg if not set by manually
# see Readme for more info
if(NOT DEFINED vcpkg_SOURCE_DIR)
  # fetch content
  include(FetchContent)

  # verbose
  set(VCPKG_VERBOSE ON)

  # Read baseline from vcpkg.json
  FetchContent_Declare(
    vcpkg
    GIT_REPOSITORY https://github.com/microsoft/vcpkg/
    GIT_TAG 2025.06.13)
  FetchContent_MakeAvailable(vcpkg)
endif()

set(CMAKE_TOOLCHAIN_FILE
  "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake"
  CACHE FILEPATH "")

# echo vcpkg_SOURCE_DIR
message(STATUS "vcpkg_SOURCE_DIR: ${vcpkg_SOURCE_DIR}")

# show which binary path we use check if  $VCPKG_DEFAULT_BINARY_CACHE
if(DEFINED ENV{VCPKG_DEFAULT_BINARY_CACHE})
  message(STATUS "VCPKG_DEFAULT_BINARY_CACHE: $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
else()
  message(STATUS "VCPKG_DEFAULT_BINARY_CACHE: not set")
endif()

# next is $XDG_CACHE_HOME/vcpkg/archives
if(DEFINED ENV{XDG_CACHE_HOME})
  message(STATUS "XDG_CACHE_HOME: $ENV{XDG_CACHE_HOME}")
else()
  message(STATUS "XDG_CACHE_HOME: not set")
endif()

# next is $HOME/.cache/vcpkg/archives
if(DEFINED ENV{HOME})
  message(STATUS "HOME: $ENV{HOME}")
  if(FLINK_CMAKE_DEBUG)
    message(
      STATUS "HOME/.cache/vcpkg/archives: $ENV{HOME}/.cache/vcpkg/archives")
    # lets check if the folder exists
    if(IS_DIRECTORY "$ENV{HOME}/.cache/vcpkg/archives")
      message(STATUS "HOME/.cache/vcpkg/archives: exists")
    else()
      message(STATUS "HOME/.cache/vcpkg/archives: does not exist")
    endif()
  endif()
else()
  message(STATUS "HOME: not set")
endif()
