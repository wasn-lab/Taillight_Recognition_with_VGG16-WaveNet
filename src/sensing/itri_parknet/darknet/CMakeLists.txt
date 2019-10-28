set(DATA_LABELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/labels)
set(DARKNET_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
configure_file(include/darknet_config.h.in
               ${PROJECT_BINARY_DIR}/include/darknet_config.h)

add_subdirectory(src)
