set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result -Wno-unknown-pragmas")
set(SRC_C_SOURCES
    activation_layer.c
    activations.c
    avgpool_layer.c
    batchnorm_layer.c
    blas.c
    box.c
    col2im.c
#    compare.c
    connected_layer.c
    convolutional_layer.c
    cost_layer.c
    crnn_layer.c
    crop_layer.c
    cuda.c
    data.c
    deconvolutional_layer.c
    demo.c
    detection_layer.c
    dropout_layer.c
    gemm.c
    gru_layer.c
    im2col.c
    image.c
    iseg_layer.c
    l2norm_layer.c
    layer.c
    list.c
    local_layer.c
    logistic_layer.c
    lstm_layer.c
    matrix.c
    maxpool_layer.c
    network.c
    normalization_layer.c
    option_list.c
    parser.c
    region_layer.c
    reorg_layer.c
    rnn_layer.c
    route_layer.c
    shortcut_layer.c
    softmax_layer.c
    tree.c
    upsample_layer.c
    utils.c
    yolo_layer.c)
set(SRC_CU_SOURCES
    activation_kernels.cu
    avgpool_layer_kernels.cu
    blas_kernels.cu
    col2im_kernels.cu
    convolutional_kernels.cu
    crop_layer_kernels.cu
    deconvolutional_kernels.cu
    dropout_layer_kernels.cu
    im2col_kernels.cu
    maxpool_layer_kernels.cu)
set(SRC_INCLUDE_DIRS
    ${DARKNET_ROOT_DIR}/include
    src
    ${PROJECT_BINARY_DIR}/include)

set_source_files_properties(utils.c PROPERTIES COMPILE_FLAGS -Wno-pointer-arith)

if (USE_CUDA)
  list(APPEND SRC_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE})
  add_definitions(-DGPU)
  include_directories(${SRC_INCLUDE_DIRS})
  cuda_add_library(darknet STATIC
                   ${SRC_C_SOURCES}
                   ${SRC_CU_SOURCES})
else ()
  include_directories(${SRC_INCLUDE_DIRS})
  add_library(darknet STATIC ${SRC_C_SOURCES})
endif()
