#include <pcl/io/lzf.h>
#include <glog/logging.h>
#include <snappy.h>
#include <zlib.h>

#include "itri_pcd_writer.h"
#include "pc2_compression_format.h"

int pc2_compressor::ITRIPCDWriter::writeBinaryCompressed(std::ostream& os, const pcl::PCLPointCloud2& cloud,
                                                         const int32_t fmt, const Eigen::Vector4f& origin,
                                                         const Eigen::Quaternionf& orientation)
{
  CHECK(!cloud.data.empty());
  if (generateHeaderBinaryCompressed(os, cloud, origin, orientation))
  {
    return -1;
  }

  std::size_t fsize = 0;
  std::size_t data_size = 0;
  std::size_t nri = 0;
  std::vector<pcl::PCLPointField> fields(cloud.fields.size());
  std::vector<int> fields_sizes(cloud.fields.size());  // number of bytes for each field
  // Compute the total size of the fields
  for (const auto& field : cloud.fields)
  {
    if (field.name == "_")
      continue;

    fields_sizes[nri] = field.count * pcl::getFieldSize(field.datatype);
    fsize += fields_sizes[nri];
    fields[nri] = field;
    ++nri;
  }
  fields_sizes.resize(nri);
  fields.resize(nri);

  // Compute the size of data
  const int32_t num_points = cloud.width * cloud.height;
  data_size = num_points * fsize;

  // If the data is too large the two 32 bit integers used to store the
  // compressed and uncompressed size will overflow.
  CHECK(data_size * 3 / 2 < std::numeric_limits<std::uint32_t>::max())
      << "The input data exceeds the allowed maximum size: " << data_size;

  //////////////////////////////////////////////////////////////////////
  // Empty array holding only the valid data
  // data_size = nr_points * point_size
  //           = nr_points * (sizeof_field_1 + sizeof_field_2 + ... sizeof_field_n)
  //           = sizeof_field_1 * nr_points + sizeof_field_2 * nr_points + ... sizeof_field_n * nr_points
  std::vector<char> only_valid_data(data_size);

  // Convert the XYZRGBXYZRGB structure to XXYYZZRGBRGB to aid compression. For
  // this, we need a vector of fields.size () (4 in this case), which points to
  // each individual plane:
  //   pters[0] = &only_valid_data[offset_of_plane_x];
  //   pters[1] = &only_valid_data[offset_of_plane_y];
  //   pters[2] = &only_valid_data[offset_of_plane_z];
  //   pters[3] = &only_valid_data[offset_of_plane_RGB];
  //
  std::vector<char*> pters(fields.size());
  std::size_t toff = 0;
  for (std::size_t i = 0; i < pters.size(); ++i)
  {
    pters[i] = &only_valid_data[toff];
    toff += fields_sizes[i] * num_points;
  }

  // Go over all the points, and copy the data in the appropriate places
  for (int32_t i = 0; i < num_points; i++)
  {
    const auto base_idx = i * cloud.point_step;
    for (std::size_t j = 0; j < pters.size(); ++j)
    {
      memcpy(pters[j], &cloud.data[base_idx + fields[j].offset], fields_sizes[j]);
      // Increment the pointer
      pters[j] += fields_sizes[j];
    }
  }

  const uint32_t temp_buf_size = data_size * 3 / 2;
  // first 8 bytes store cmpr/decmpr size information.
  std::vector<char> temp_buf(temp_buf_size + 8);

  // Compress the valid data

  uint32_t compressed_size = 0;
  if (fmt == compression_format::lzf)
  {
    compressed_size =
        pcl::lzfCompress(&only_valid_data.front(), static_cast<unsigned int>(data_size), &temp_buf[8], temp_buf_size);
  }
  else if (fmt == compression_format::snappy)
  {
    size_t cmpr_size;
    snappy::RawCompress(&only_valid_data.front(), data_size, &temp_buf[8], &cmpr_size);
    compressed_size = cmpr_size;
  }
  else if (fmt == compression_format::none)
  {
    compressed_size = static_cast<unsigned int>(data_size);
    memcpy(&temp_buf[8], &only_valid_data.front(), compressed_size);
  }
  else if (fmt == compression_format::zlib)
  {
    uint64_t cmpr_size = temp_buf_size;
    const int cmpr_level = 1;  // 0: no compression, 1: best speed, 9: best compression
    compress2(reinterpret_cast<unsigned char*>(&temp_buf[8]), &cmpr_size,
              reinterpret_cast<const unsigned char*>(&only_valid_data.front()), data_size, cmpr_level);
    compressed_size = cmpr_size;
  }
  else
  {
    CHECK(false) << "Unaccepted compression format: " << fmt;
    return -1;
  }

  if (compressed_size == 0)
  {
    return -1;
  }

  memcpy(&temp_buf[0], &compressed_size, 4);
  memcpy(&temp_buf[4], &data_size, 4);
  temp_buf.resize(compressed_size + 8);

  os.imbue(std::locale::classic());
  os << "DATA binary_compressed\n";
  std::copy(temp_buf.begin(), temp_buf.end(), std::ostream_iterator<char>(os));
  os.flush();

  return (os ? 0 : -1);
}
