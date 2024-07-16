/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include <cudaTypedefs.h>

int nccl_net_ofi_cuda_init(void);

/*
 * @brief	Gets the CUDA device associated with the buffer
 *
 * @param	data
 *		Pointer to CUDA buffer.
 *
 * @return	Valid CUDA device ID on success
 *		-1 on error
 * @return	0 on success
 *		non-zero on error
 */
int nccl_net_ofi_get_cuda_device(void *data, int *dev_id);

extern PFN_cuDriverGetVersion nccl_net_ofi_cuDriverGetVersion;
extern PFN_cuPointerGetAttribute nccl_net_ofi_cuPointerGetAttribute;
extern PFN_cuCtxGetDevice nccl_net_ofi_cuCtxGetDevice;
extern PFN_cuDeviceGetCount nccl_net_ofi_cuDeviceGetCount;

#if CUDA_VERSION >= 11030
extern PFN_cuFlushGPUDirectRDMAWrites nccl_net_ofi_cuFlushGPUDirectRDMAWrites;
#else
extern void *nccl_net_ofi_cuFlushGPUDirectRDMAWrites;
#endif

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
