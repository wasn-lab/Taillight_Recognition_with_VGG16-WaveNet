include(CompilerFlags)
cmake_policy(SET CMP0043 OLD)
cmake_policy(SET CMP0046 OLD)

option(ENABLE_CCACHE "Use ccache to speed up build process" ON)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND ENABLE_CCACHE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

