include(CompilerFlags)
cmake_policy(SET CMP0043 OLD)
cmake_policy(SET CMP0046 OLD)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND NOT SCAN_BUILD_MODE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

