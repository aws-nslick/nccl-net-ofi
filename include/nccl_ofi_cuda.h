/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_


#ifdef __cplusplus
extern "C" {
#endif

int nccl_net_ofi_cuda_init(void) __attribute__((weak));

/*
 * @brief	Gets the CUDA device associated with the buffer
 *
 * @param	data
 *		Pointer to CUDA buffer.
 *
 * @return	Valid CUDA device ID on success
 *		-1 on error
 * @return	0 on success
 *		-EINVAL on error
 */
int nccl_net_ofi_get_cuda_device_for_addr(void *data, int *dev_id)  __attribute__((weak));

/*
 * @brief	wraps cudaFlushGPUDirectRDMAWrites() with default args.

 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_flush(void)  __attribute__((weak));

/*
 * @brief	wraps cudaGetDevice()

 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_get_num_devices(void)  __attribute__((weak));

/*
 * @brief	wraps cudaGetDeviceCount()

 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_get_active_device_idx(void)  __attribute__((weak));

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
