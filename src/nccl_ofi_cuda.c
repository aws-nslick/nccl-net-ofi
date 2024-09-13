/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"
#include <errno.h>
#include <cuda_runtime_api.h>
#include "nccl_ofi.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_log.h"

int nccl_net_ofi_cuda_flush(void)
{
#ifdef __cplusplus
	using cudaFlushGPUDirectRDMAWritesTarget::*;
	using cudaFlushGPUDirectRDMAWritesScope::*;
#endif
		cudaError_t ret = cudaDeviceFlushGPUDirectRDMAWrites(
			cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
			cudaFlushGPUDirectRDMAWritesToOwner);
		return (ret == cudaSuccess) ? 0 : -EPERM;
}

int nccl_net_ofi_cuda_init(void)
{
	int driverVersion = -1;
	int runtimeVersion = -1;

	{
		cudaError_t res = cudaDriverGetVersion(&driverVersion);
		if (res != cudaSuccess) {
			NCCL_OFI_WARN("Failed to query CUDA driver version.");
			return -EINVAL;
		}
	}

	{
		cudaError_t res = cudaRuntimeGetVersion(&driverVersion);
		if (res != cudaSuccess) {
			NCCL_OFI_WARN("Failed to query CUDA runtime version.");
			return -EINVAL;
		}
	}


	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using CUDA driver version %d with runtime %d", driverVersion, runtimeVersion);

	if (ofi_nccl_cuda_flush_enable()) {
		NCCL_OFI_WARN("CUDA flush enabled");
		cuda_flush = true;
	} else {
		cuda_flush = false;
	}

	return 0;
}


int nccl_net_ofi_cuda_get_num_devices(void)
{
	int count = -1;
	cudaError_t res = cudaGetDeviceCount(&count);
	return res == cudaSuccess ? count : -1;
}

int nccl_net_ofi_cuda_get_active_device_idx(void)
{
	int index = -1;
	cudaError_t res = cudaGetDevice(&index);
	return res == cudaSuccess ? index : -1;
}


int nccl_net_ofi_get_cuda_device_for_addr(void *data, int *dev_id)
{
	struct cudaPointerAttributes attrs = {};
	cudaError_t res = cudaPointerGetAttributes(&attrs, data);
	if (res != cudaSuccess)
		return -EINVAL;

	switch (attrs.type) {
		case cudaMemoryTypeDevice:
			*dev_id = attrs.device;
			return 0;
		default:
			NCCL_OFI_WARN("Invalid buffer pointer provided");
			*dev_id = -1;
			return -EINVAL;
	};
}

