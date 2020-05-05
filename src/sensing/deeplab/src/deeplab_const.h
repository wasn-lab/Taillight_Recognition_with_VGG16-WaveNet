#pragma once

namespace deeplab {
constexpr int32_t image_width = 513;
constexpr int32_t image_height = 513;
constexpr int32_t num_pixels = image_width * image_height;
constexpr int32_t bytes_per_pixel = 3;
constexpr int32_t input_tensor_size_in_bytes = num_pixels * bytes_per_pixel * sizeof(uint8_t);

};  // namespace deeplab
