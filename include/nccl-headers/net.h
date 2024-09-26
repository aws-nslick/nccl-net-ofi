/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_HEADERS_NET_H
#define NCCL_HEADERS_NET_H

// IWYU pragma: begin_exports
#if HAVE_CUDA
#include "nccl-headers/nvidia/net.h"
#elif HAVE_NEURON
#include "nccl-headers/neuron/net.h"
#else
#error "Neither CUDA nor Neuron support is available"
#endif
// IWYU pragma: end_exports

#endif
