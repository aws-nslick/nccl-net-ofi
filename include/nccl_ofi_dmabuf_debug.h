#ifndef NCCL_OFI_DMABUF_DEBUG_H_
#define NCCL_OFI_DMABUF_DEBUG_H_

#include "config.h"

#include <alloca.h>
#include <linux/dma-buf.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>

#include "nccl_ofi_mr.h"
#include "nccl_ofi_param.h"

static inline void nccl_ofi_add_dmabuf_debug_label(nccl_ofi_mr_ckey_ref ckey, const char *device_name)
{
#if HAVE_DECL_FI_MR_DMABUF
	if (ckey->type != NCCL_OFI_MR_CKEY_DMABUF)
		return;

	const bool enabled = (bool)ofi_nccl_debug_dmabuf_labels();
	if (enabled) {
		char *str = (char *)alloca(DMA_BUF_NAME_LEN);
		snprintf(str, DMA_BUF_NAME_LEN, "nnofi::%s", device_name);
		// Nothing we can do if it failed, ignore it.
		(void)ioctl(ckey->fi_mr_dmabuf.fd, DMA_BUF_SET_NAME, str);
	}
#endif
}


#endif  // NCCL_OFI_DMABUF_DEBUG_H_
