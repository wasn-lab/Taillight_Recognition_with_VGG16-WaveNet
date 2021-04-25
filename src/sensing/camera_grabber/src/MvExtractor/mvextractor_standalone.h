/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __MVEXTRACTOR_STANDALONE_H__
#define __MVEXTRACTOR_STANDALONE_H__

#include <opencv2/opencv.hpp>
#include "NvVideoEncoder.h"
#include <linux/videodev2.h>
#include <malloc.h>
#include <string.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <semaphore.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <future>
#include "SafeQueue.hpp"

#define MAX_MVS_QUEUE_SIZE   20
#define MAX_OUTPUT_PLANE_FD  32

typedef struct
{
    uint32_t                        frame_number; // capture plane dequeue frame number
    uint32_t                        total_numbers;
    std::vector<uint32_t>           mvs;
} MotionVectors;

typedef struct
{
    NvVideoEncoder *enc;
    uint32_t encoder_pixfmt;
    uint32_t raw_pixfmt;
    uint32_t width;
    uint32_t height;
    uint32_t bitrate;
    uint32_t peak_bitrate;
    uint32_t profile;
    uint32_t iframe_interval;
    uint32_t idr_interval;
    uint32_t level;
    uint32_t fps_n;
    uint32_t fps_d;
    uint32_t num_b_frames;
    uint32_t num_reference_frames;
    uint32_t num_output_buffers;
    std::ofstream *out_file;
    enum v4l2_mpeg_video_bitrate_mode ratecontrol;
    int output_plane_fd[MAX_OUTPUT_PLANE_FD];

    bool stats;
    bool got_error;
    bool copy_timestamp;
    uint32_t start_ts;
    uint64_t timestamp;
    uint64_t timestampincr;

    int encoder_index;
    int encoder_capture_plane_dq_num;
    std::string encoder_name;

    MotionVectors motion_vectors;
    SafeQueue<MotionVectors> mvs_queue;

} context_t;

void MvExtractor_Settings(context_t *ctx);
int MvExtractor_Init(context_t *ctx, int index);
int MvExtractor_Process(context_t *ctx, cv::Mat *cvYUV);
int MvExtractor_Deinit(context_t *ctx);

#endif 