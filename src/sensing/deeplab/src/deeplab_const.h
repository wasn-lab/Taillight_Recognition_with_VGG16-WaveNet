#pragma once

namespace deeplab
{
constexpr int32_t DEEPLAB_IMAGE_WIDTH = 513;
constexpr int32_t DEEPLAB_IMAGE_HEIGHT = 513;
constexpr int32_t NUM_PIXELS = DEEPLAB_IMAGE_WIDTH * DEEPLAB_IMAGE_HEIGHT;
constexpr int32_t BYTES_PER_PIXEL = 3;
constexpr int32_t INPUT_TENSOR_SIZE_IN_BYTES = NUM_PIXELS * BYTES_PER_PIXEL * sizeof(uint8_t);

constexpr int32_t ROS_IMAGE_WIDTH = 608;
constexpr int32_t ROS_IMAGE_HEIGHT = 384;

// given 608x384, pad it into 608x608, shink to 513x513 for inference.
// When publish, covert 513x513 back to 608x384
constexpr int32_t PADDING_IN_PIXELS = ROS_IMAGE_WIDTH - ROS_IMAGE_HEIGHT;
constexpr int32_t PADDING_TOP_IN_PIXELS = PADDING_IN_PIXELS / 2;
constexpr int32_t PADDING_BOTTOM_IN_PIXELS = PADDING_IN_PIXELS - PADDING_TOP_IN_PIXELS;

};  // namespace deeplab
