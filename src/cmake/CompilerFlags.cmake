# Unified setting for compiler flags.

option(USE_GPROF "Use gprof for performance profiling" OFF)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(COMPILER_IS_CLANG TRUE)
    set(COMPILER_IS_GNUCXX FALSE)
endif ()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(COMPILER_IS_CLANG FALSE)
    set(COMPILER_IS_GNUCXX TRUE)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR COMPILER_IS_CLANG)
    set(COMPILER_IS_GCC_OR_CLANG ON)
endif ()

set(CPU_IS_X86 FALSE)
set(CPU_IS_ARM FALSE)
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CPU_IS_X86 TRUE)
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(CPU_IS_ARM TRUE)
else ()
  message(FATAL_ERROR "Cannot decide cpu type")
endif ()

set(CMAKE_CXX_STANDARD 14)  # -std=c++14
set(CMAKE_C_STANDARD 11)  # -std=gnu11
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC -fPIE

set(CUDA_NVCC_FLAGS
    -O3
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
    -gencode=arch=compute_70,code=sm_70
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_75,code=compute_75
    --default-stream per-thread
    --keep  # keep intermediate files for generating coverage reports.
)

macro(APPEND_GLOBAL_COMPILER_FLAGS)
    foreach (_flag ${ARGN})
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_flag}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_flag}")
    endforeach ()
endmacro()

macro(ENABLE_SANITIZER sanitizer_t)
    APPEND_GLOBAL_COMPILER_FLAGS(-fno-omit-frame-pointer
                                 -fno-optimize-sibling-calls)
    set(SANITIZER_FLAGS "-fsanitize=${sanitizer_t}")
#    if (COMPILER_IS_GNUCXX)
#      if (${sanitizer_t} STREQUAL "address")
#        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -static-libasan")
#      elseif (${sanitizer_t} STREQUAL "thread")
#        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -static-libtsan")
#      elseif (${sanitizer_t} STREQUAL "leak")
#        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -static-liblsan")
#      elseif (${sanitizer_t} STREQUAL "undefined")
#        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -static-libubsan")
#      endif ()
#    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS
        "-lpthread ${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS
        "-lpthread ${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS
        "-lpthread ${CMAKE_MODULE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
endmacro()

# Warning messages:
# yolo_src contains too many warnings, so full warnings only apply to *.cpp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror=reorder")

APPEND_GLOBAL_COMPILER_FLAGS(
    -D_FORTIFY_SOURCE=2
    -Wno-deprecated-declarations
    -Wno-comment
    -Wno-unused-parameter
    -Wno-deprecated-declarations
#    -Wcast-align
    -Wformat-security
    -Wpointer-arith
    -Wwrite-strings)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result")
if (COMPILER_IS_GNUCXX)
    APPEND_GLOBAL_COMPILER_FLAGS(-Wmaybe-uninitialized)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-discarded-qualifiers")
endif ()

if (DEFINED SCAN_BUILD_MODE)
  APPEND_GLOBAL_COMPILER_FLAGS(-U_FORTIFY_SOURCE)
endif()

if (COMPILER_IS_CLANG)
    APPEND_GLOBAL_COMPILER_FLAGS(-Wuninitialized)
    APPEND_GLOBAL_COMPILER_FLAGS(-Wno-missing-braces)
endif ()


if (ENABLE_ADDRESS_SANITIZER)
    ENABLE_SANITIZER("address")
endif ()

if (ENABLE_THREAD_SANITIZER)
    ENABLE_SANITIZER("thread")
endif ()

if (ENABLE_LEAK_SANITIZER)
    ENABLE_SANITIZER("leak")
endif ()

if (ENABLE_UNDEFINED_SANITIZER)
    ENABLE_SANITIZER("undefined")
endif ()

if (ENABLE_GCOV AND CMAKE_COMPILER_IS_GNUCXX)
    include(CodeCoverage)
    APPEND_COVERAGE_COMPILER_FLAGS()
    set(COVERAGE_LCOV_EXCLUDES "boost/*" "c++/*" "gtest/*")
endif ()

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (USE_GPROF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif ()
