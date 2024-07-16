/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"
#define _GNU_SOURCE
#include <link.h>
#include <dlfcn.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "nccl_ofi_cuda.h"
#include "nccl_ofi_log.h"

typedef cudaError_t (*runtimeFnResolverFn_t)(const char*, void**, unsigned long long, void*);

PFN_cuDriverGetVersion nccl_net_ofi_cuDriverGetVersion = NULL;
PFN_cuPointerGetAttribute nccl_net_ofi_cuPointerGetAttribute = NULL;
PFN_cuCtxGetDevice nccl_net_ofi_cuCtxGetDevice = NULL;
PFN_cuDeviceGetCount nccl_net_ofi_cuDeviceGetCount = NULL;

#if CUDA_VERSION >= 11030
PFN_cuFlushGPUDirectRDMAWrites nccl_net_ofi_cuFlushGPUDirectRDMAWrites = NULL;
#else
void *nccl_net_ofi_cuFlushGPUDirectRDMAWrites = NULL;
#endif


#define STRINGIFY(sym) # sym


/* Blindly load a symbol from lib */
#define LOAD_DIRECT_SYM(lib, sym) do {									\
	nccl_net_ofi_##sym = (typeof(sym) *)dlsym(lib, STRINGIFY(sym));		\
	if (nccl_net_ofi_##sym == NULL) {									\
		NCCL_OFI_WARN("Failed to load symbol via dlsym: "				\
					  STRINGIFY(sym));									\
		return -ENOTSUP;												\
	}} while(0)

/* Using cuGetProcAddress, load sym at api_ver. */
#define LOAD_DRIVER_VERSIONED_SYM(cuGetProcAddress, api_ver, sym) do {	\
	CUresult status = cuGetProcAddress(									\
		STRINGIFY(sym),													\
		(void**) nccl_net_ofi_ ## sym,									\
		api_ver, CU_GET_PROC_ADDRESS_DEFAULT, NULL);					\
																		\
	if (status != CUDA_SUCCESS) {										\
		NCCL_OFI_WARN("Failed to load symbol via cuGetProcAddress: "	\
					  STRINGIFY(sym));									\
		return -ENOTSUP;												\
	}} while(0)

/* Using cudaGetDriverEntryPoint, load implicitly versioned sym. */
#define LOAD_RUNTIME_VERSIONED_SYM(resolve, sym) do {					\
	cudaError_t status = resolve(									    \
		STRINGIFY(sym),													\
		(void**)nccl_net_ofi_##sym,										\
		cudaEnableDefault, NULL											\
    );										                            \
	if (status != cudaSuccess) {										\
		NCCL_OFI_WARN(													\
			"Failed to load symbol via cudaGetDriverEntryPoint: "		\
			STRINGIFY(sym));											\
		return -ENOTSUP;												\
	}} while(0)

static int find_cudart_dso(struct dl_phdr_info *info, size_t size, void *data)
{
	Dl_info addr_info = {};
	struct link_map map = {};
	if (strstr(info->dlpi_name, "libcudart.so") == NULL)
		return 0;

	int status = dladdr1((void*)info->dlpi_addr, &addr_info, (void**)&map, RTLD_DL_LINKMAP);
	if (status != 0) {
		return 1;
	}

	void *handle = dlopen(map.l_name, RTLD_LOCAL | RTLD_NOW);
	if (handle != NULL) {
		*((void**)data) = dlsym(handle, "cudaGetDriverEntryPoint");
	}

	dlclose(handle);
	return 1;
}

static int
load_syms_runtime(runtimeFnResolverFn_t resolve)
{
	assert(resolve != NULL);
	LOAD_RUNTIME_VERSIONED_SYM(resolve, cuDriverGetVersion);
	LOAD_RUNTIME_VERSIONED_SYM(resolve, cuPointerGetAttribute);
	LOAD_RUNTIME_VERSIONED_SYM(resolve, cuCtxGetDevice);
	LOAD_RUNTIME_VERSIONED_SYM(resolve, cuDeviceGetCount);
#if CUDA_VERSION >= 11030
	LOAD_RUNTIME_VERSIONED_SYM(resolve, cuFlushGPUDirectRDMAWrites);
#endif
	return 0;
}


static int
load_syms_driver(void)
{
	/* Clear any previous errors */
	(void) dlerror();

	// the driver stub from the runtime ships as `libcuda.so', so prefer
	// the more specific `libcuda.so.1' here to avoid loading it.
	void *cudadriver_lib = dlopen("libcuda.so.1", RTLD_NOW);
	if (cudadriver_lib == NULL) {
		NCCL_OFI_WARN("Failed to find CUDA Driver library: %s", dlerror());
		return -ENOTSUP;
	}

	PFN_cuCtxGetCurrent nccl_net_ofi_cuCtxGetCurrent = NULL;
	PFN_cuCtxGetApiVersion nccl_net_ofi_cuCtxGetApiVersion = NULL;
	PFN_cuGetProcAddress nccl_net_ofi_cuGetProcAddress = NULL;

	LOAD_DIRECT_SYM(cudadriver_lib, cuCtxGetCurrent);
	LOAD_DIRECT_SYM(cudadriver_lib, cuCtxGetApiVersion);
	LOAD_DIRECT_SYM(cudadriver_lib, cuGetProcAddress);

	CUcontext ctx = NULL;
	CUresult result = nccl_net_ofi_cuCtxGetCurrent(&ctx);
	if (result != CUDA_SUCCESS) {
		NCCL_OFI_WARN("Failed to get current CUDA Driver context");
		return -ENOTSUP;
	}

	unsigned int context_version = 0;
	result = nccl_net_ofi_cuCtxGetApiVersion(ctx, &context_version);
	if (result != CUDA_SUCCESS) {
		NCCL_OFI_WARN("Failed to get current CUDA Driver context version");
		return -ENOTSUP;
	}

	LOAD_DRIVER_VERSIONED_SYM(nccl_net_ofi_cuGetProcAddress, context_version, cuDriverGetVersion);
	LOAD_DRIVER_VERSIONED_SYM(nccl_net_ofi_cuGetProcAddress, context_version, cuPointerGetAttribute);
	LOAD_DRIVER_VERSIONED_SYM(nccl_net_ofi_cuGetProcAddress, context_version, cuCtxGetDevice);
	LOAD_DRIVER_VERSIONED_SYM(nccl_net_ofi_cuGetProcAddress, context_version, cuDeviceGetCount);
#if CUDA_VERSION >= 11030
	LOAD_DRIVER_VERSIONED_SYM(nccl_net_ofi_cuGetProcAddress, context_version, cuFlushGPUDirectRDMAWrites);
#endif

	return 0;
}

int
nccl_net_ofi_cuda_init(void)
{
	runtimeFnResolverFn_t func = NULL;
	dl_iterate_phdr(find_cudart_dso, (void*)&func);
	if (func == NULL) {
		NCCL_OFI_WARN("libcudart.so was not found in PHDRs, falling back to loading libcuda.so.1 directly. ABI breakage possible!");
		return load_syms_driver();
	}

	return load_syms_runtime(func);
}


int nccl_net_ofi_get_cuda_device(void *data, int *dev_id)
{
	int ret = 0;
	int cuda_device = -1;
	unsigned int mem_type;
	unsigned int device_ordinal;
	CUresult cuda_ret_mem = nccl_net_ofi_cuPointerGetAttribute(&mem_type,
								   CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
								   (CUdeviceptr) data);
	CUresult cuda_ret_dev = nccl_net_ofi_cuPointerGetAttribute(&device_ordinal,	
								   CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
								   (CUdeviceptr) data);

	if (cuda_ret_mem != CUDA_SUCCESS || cuda_ret_dev != CUDA_SUCCESS) {
		ret = -ENOTSUP;
		NCCL_OFI_WARN("Invalid buffer pointer provided");
		goto exit;
	}

	if (mem_type == CU_MEMORYTYPE_DEVICE) {
		cuda_device = device_ordinal;
	} else {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid type of buffer provided. Only device memory is expected for NCCL_PTR_CUDA type");
	}

 exit:
	*dev_id = cuda_device;
	return ret;
}

