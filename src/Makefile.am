#
# Copyright (c) 2018-2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(top_srcdir)/include -DXML_DIR=\"${pkgdatadir}/xml\"

sources = \
	nccl_ofi_api.cc \
	nccl_ofi_net.cc \
	nccl_ofi_sendrecv.cc \
	nccl_ofi_rdma.cc \
	nccl_ofi_scheduler.cc \
	nccl_ofi_topo.cc \
	nccl_ofi_msgbuff.cc \
	nccl_ofi_freelist.cc \
	nccl_ofi_deque.cc \
	nccl_ofi_idpool.cc \
	nccl_ofi_ofiutils.cc \
	tracepoint.cc

if WANT_PLATFORM_AWS
sources += platform-aws.cc
endif

if ENABLE_NEURON
  lib_LTLIBRARIES = libnccom-net.la
  sources += nccl_ofi_interface_neuron.cc
  libnccom_net_la_SOURCES = $(sources)
else
  lib_LTLIBRARIES = libnccl-net.la
  sources += nccl_ofi_cuda.cc \
	nccl_ofi_interface_nvidia.cc
  libnccl_net_la_SOURCES = $(sources)

if WANT_PLATFORM_AWS
  # NCCL tuner plugin
  lib_LTLIBRARIES += libnccl-ofi-tuner.la
  tuner_sources = \
	tuner/nccl_ofi_model.cc \
	tuner/nccl_ofi_tuner.cc
  libnccl_ofi_tuner_la_SOURCES = $(tuner_sources)
endif
endif
