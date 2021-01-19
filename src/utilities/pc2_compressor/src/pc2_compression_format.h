#pragma once

namespace pc2_compressor
{
/*
                cmpr time       decmpr time       cmpr ratio       cmpr ratio
                   (ms)             (ms)        (velodyne 32)      (ouster 64)
  lzf               18              1.5             0.45             0.33
  snappy            13              0.8             0.44             0.33
  zlib (level=1)    41              5.8             0.39             0.26
  zlib (level=9)    58              5.8             0.37             0.25

  Time was measure at Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
*/
enum compression_format
{
  none,  // no compression
  lzf,
  snappy,
  zlib,
  nums
};

};  // namespace pc2_compressor
