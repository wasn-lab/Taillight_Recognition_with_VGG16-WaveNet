/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */
#include <pcl/io/lzf.h>
#include <glog/logging.h>
#include <snappy.h>
#include <zlib.h>
#include "itri_pcd_reader.h"
#include "pc2_compression_format.h"

int pc2_compressor::ITRIPCDReader::readBodyCompressed(const unsigned char* data_cmpr, pcl::PCLPointCloud2& cloud,
                                                      const int32_t fmt, const uint32_t data_idx)
{
  // Setting the is_dense property to true by default
  cloud.is_dense = 1u;

  unsigned int compressed_size = 0, uncompressed_size = 0;
  memcpy(&compressed_size, &data_cmpr[data_idx + 0], 4);
  memcpy(&uncompressed_size, &data_cmpr[data_idx + 4], 4);

  if (uncompressed_size != cloud.data.size())
  {
    cloud.data.resize(uncompressed_size);
  }

  auto data_size = static_cast<unsigned int>(cloud.data.size());
  std::vector<char> buf(data_size);
  // The size of the uncompressed data better be the same as what we stored in the header
  if (fmt == compression_format::lzf)
  {
    unsigned int tmp_size = pcl::lzfDecompress(&data_cmpr[data_idx + 8], compressed_size, &buf[0], data_size);
    CHECK(tmp_size == uncompressed_size);
  }
  else if (fmt == compression_format::snappy)
  {
    bool success =
        snappy::RawUncompress(reinterpret_cast<const char*>(&data_cmpr[data_idx + 8]), compressed_size, &buf[0]);
    CHECK(success == true);
  }
  else if (fmt == compression_format::none)
  {
    memcpy(&buf[0], reinterpret_cast<const char*>(&data_cmpr[data_idx + 8]), uncompressed_size);
  }
  else if (fmt == compression_format::zlib)
  {
    uint64_t dest_len = uncompressed_size;
    auto ret = uncompress(reinterpret_cast<unsigned char*>(&buf[0]), &dest_len,
                          reinterpret_cast<const unsigned char*>(&data_cmpr[data_idx + 8]), compressed_size);
    CHECK(ret == Z_OK) << "uncompress returns " << ret;
    CHECK(dest_len == uncompressed_size);
  }
  else
  {
    CHECK(false) << "Unaccepted compression format: " << fmt;
    return -1;
  }

  // Get the fields sizes
  std::vector<pcl::PCLPointField> fields(cloud.fields.size());
  std::vector<int> fields_sizes(cloud.fields.size());
  int nri = 0, fsize = 0;
  for (const auto& field : cloud.fields)
  {
    if (field.name == "_")
    {
      continue;
    }
    fields_sizes[nri] = field.count * pcl::getFieldSize(field.datatype);
    fsize += fields_sizes[nri];
    fields[nri] = field;
    ++nri;
  }
  fields.resize(nri);
  fields_sizes.resize(nri);

  // Unpack the xxyyzz to xyz
  const int32_t num_points = cloud.width * cloud.height;
  std::vector<char*> pters(fields.size());
  std::size_t toff = 0;
  for (std::size_t i = 0; i < pters.size(); ++i)
  {
    pters[i] = &buf[toff];
    toff += fields_sizes[i] * num_points;
  }
  // Copy it to the cloud
  for (int32_t i = 0; i < num_points; i++)
  {
    const auto base_idx = i * fsize;
    for (std::size_t j = 0; j < pters.size(); ++j)
    {
      memcpy(&cloud.data[base_idx + fields[j].offset], pters[j], fields_sizes[j]);
      // Increment the pointer
      pters[j] += fields_sizes[j];
    }
  }
  return 0;
}
