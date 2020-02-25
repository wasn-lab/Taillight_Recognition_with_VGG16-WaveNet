set(CAR_MODEL_IS_B1 0)
set(CAR_MODEL_IS_B1V2 0)
set(CAR_MODEL_IS_C 0)
set(CAR_MODEL_IS_HINO 0)

if ("${CAR_MODEL}" STREQUAL "B1")
  set(CAR_MODEL_IS_B1 1)
elseif ("${CAR_MODEL}" STREQUAL "B1V2")
  set(CAR_MODEL_IS_B1V2 1)
elseif ("${CAR_MODEL}" STREQUAL "C")
  set(CAR_MODEL_IS_C 1)
elseif ("${CAR_MODEL}" STREQUAL "HINO")
  set(CAR_MODEL_IS_HINO 1)
else ()
  message(FATAL_ERROR "Invalid car model: ${CAR_MODEL}")
endif ()
