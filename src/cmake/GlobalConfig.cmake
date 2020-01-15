if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to Release as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(CompilerFlags)
cmake_policy(SET CMP0043 OLD)
cmake_policy(SET CMP0046 OLD)

option(ENABLE_CCACHE "Use ccache to speed up build process" ON)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND ENABLE_CCACHE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

include(DebPack)
