find_package(PythonInterp REQUIRED)
set(module em)
string(TOUPPER ${module} module_upper)

if(NOT PY_${module_upper})
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c" "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
        RESULT_VARIABLE _${module}_status
        OUTPUT_VARIABLE _${module}_location
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _${module}_status)
        set(PY_${module_upper} ${_${module}_location} CACHE
            STRING "Find Python module ${module}.")
    else()
        MESSAGE(FATAL_ERROR "empy not found, try something like: sudo apt-get install empy")
    endif(NOT _${module}_status)
    find_package_handle_standard_args(PY_${module} DEFAULT_MSG PY_${module_upper})
endif(NOT PY_${module_upper})
