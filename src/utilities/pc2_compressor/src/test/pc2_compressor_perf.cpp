/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <benchmark/benchmark.h>
#include "pc2_compressor_test_utils.h"

static void BM_pc2_cmpr_lzf(benchmark::State& state)
{
  // Perform setup here
  gen_rand_cloud();
  for (auto _ : state)
  {
    auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::lzf);
  }
}

static void BM_pc2_decmpr_lzf(benchmark::State& state)
{
  // Perform setup here
  gen_rand_cloud();
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::lzf);
  for (auto _ : state)
  {
    auto decmpr_ros_pc2_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  }
}

static void BM_pc2_cmpr_snappy(benchmark::State& state)
{
  // Perform setup here
  gen_rand_cloud();
  for (auto _ : state)
  {
    auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::snappy);
  }
}

static void BM_pc2_decmpr_snappy(benchmark::State& state)
{
  // Perform setup here
  gen_rand_cloud();
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::snappy);
  for (auto _ : state)
  {
    auto decmpr_ros_pc2_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  }
}

static void BM_pc2_cmpr_zlib(benchmark::State& state)
{
  // Perform setup here
  gen_rand_cloud();
  for (auto _ : state)
  {
    auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::zlib);
  }
}

static void BM_pc2_decmpr_zlib(benchmark::State& state)
{
  // Perform setup here
  gen_rand_cloud();
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::zlib);
  for (auto _ : state)
  {
    auto decmpr_ros_pc2_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  }
}


// Register the function as a benchmark
BENCHMARK(BM_pc2_cmpr_lzf);
BENCHMARK(BM_pc2_decmpr_lzf);
BENCHMARK(BM_pc2_cmpr_snappy);
BENCHMARK(BM_pc2_decmpr_snappy);
BENCHMARK(BM_pc2_cmpr_zlib);
BENCHMARK(BM_pc2_decmpr_zlib);
// Run the benchmark
BENCHMARK_MAIN();
