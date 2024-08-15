#
# Copyright (c) 2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
#
# Usage: https://docs.docker.com/reference/cli/docker/buildx/bake/

# Notes:
# * arm64 builds will use qemu by default, but requires containerd snapshotting
#   to be enabled in docker's daemon.json, or explicit creation of an arm64
#   capable context.
#
# * developers should strongly consider standing up an eks cluster and
#   configuring a k8s builder for native arm64 builds:
#    https://docs.docker.com/build/builders/drivers/kubernetes/

group "default" { targets = [ "rpms", "debs" ] }

variable "base_images" {
  type    = list(string)
  default = [
      "amazonlinux:2",
      "amazonlinux:2023",
      "rockylinux:8",
      "rockylinux:9",
      "opensuse/leap:15",
      # Intentionaly not included
      # "centos:centos7"
      # Debian
      "debian:10",
      "debian:11",
      # "debian:12" # not supported by EFA installer.
      "ubuntu:20.04",
      "ubuntu:22.04",
      "ubuntu:24.04",
  ]
}


function "efa_installer_dir_name" {
  params = [base_image]
  result = "${replace(replace(replace(replace(upper(replace(replace(base_image, ".", ""), ":", "")), "MAZON", ""), "/", ""), "OPEN", ""), "LEAP15", "")}"
}

function "plaintext_image_name" {
  params = [base_image]
  result = "${replace(replace(replace(base_image, "/", "_"), ":", ""), ".", "")}"
}

function "baseimage_to_cuda_repo_name" {
  params = [base_image]
  result = "${replace(replace(base_image, ":", ""), ".", "")}"
}

# Caches efa installer packages, without actually installing them.
target "efa_installer_base_images" {
  name = "${plaintext_image_name(base_image)}_base_efa-installer-${replace(efa_installer_version, ".", "-")}${debug_enabled == 1 ? "-debugsyms" : ""}${ item.mpi4_enabled == 1 ? "-mpi4" : "" }${ item.mpi5_enabled == 1 ? "-mpi5" : ""}"
  tags = [ "982534352369.dkr.ecr.us-west-1.amazonaws.com/common/efa_installer/${plaintext_image_name(base_image)}${debug_enabled == 1 ? "-debugsyms" : ""}${ item.mpi4_enabled == 1 ? "-mpi4" : "" }${ item.mpi5_enabled == 1 ? "-mpi5" : ""}:${efa_installer_version}" ]
  matrix = {
    base_image = base_images,
    #  "amazonlinux:2",
    #  "amazonlinux:2023",
    #  "rockylinux:8",
    #  "rockylinux:9",
    #  "opensuse/leap:15",
    #  # Intentionaly not included
    #  # "centos:centos7"
    #  # Debian
    #  "debian:10",
    #  "debian:11",
    #  # "debian:12" # not supported by EFA installer.
    #  "ubuntu:20.04",
    #  "ubuntu:22.04",
    #  "ubuntu:24.04",
    #]
    efa_installer_version = [
      "1.34.0",
      "1.33.0",
    ]
    item = [
      { mpi4_enabled = 1, mpi5_enabled = 0 },
      { mpi4_enabled = 0, mpi5_enabled = 1 },
      { mpi4_enabled = 0, mpi5_enabled = 0 },
    ]
    debug_enabled = [ 0, 1 ]
  }
  contexts = {
    efa_installer_tarball = "https://efa-installer.amazonaws.com/aws-efa-installer-${efa_installer_version}.tar.gz"
    distro_image = "docker-image://${base_image}"
  }
  targets = [ "linux/amd64", "linux/arm64" ]
  args = {
    INSTALLER_PREFIX = "${efa_installer_dir_name(base_image)}"
    ENABLE_EFA_INSTALLER_DEBUG_INFO = debug_enabled
    ENABLE_MPI4 = item.mpi4_enabled,
    ENABLE_MPI5 = item.mpi5_enabled,
  }
  dockerfile = ".docker/containers/Dockerfile.cache_efa"
  #output = ["type=image,push=true"]
}

target "cuda_enabled_build_images" {
  name = "${plaintext_image_name(base_image)}_efa-installer-${replace(efa_installer_version, ".", "-")}${debug_enabled == 1 ? "-debugsyms" : ""}${ item.mpi4_enabled == 1 ? "-mpi4" : "" }${ item.mpi5_enabled == 1 ? "-mpi5" : ""}"
  tags = [ "982534352369.dkr.ecr.us-west-1.amazonaws.com/common/efa_installer/${plaintext_image_name(base_image)}${debug_enabled == 1 ? "-debugsyms" : ""}${ item.mpi4_enabled == 1 ? "-mpi4" : "" }${ item.mpi5_enabled == 1 ? "-mpi5" : ""}:${efa_installer_version}-cuda${cuda_version}" ]
  dockerfile = ".docker/containers/Dockerfile.dpkg_add_cuda_repo"
  output = ["type=cacheonly"]
  args = { CUDA_DISTRO = "${baseimage_to_cuda_repo_name(base_image)}", CUDA_TOOLKIT_VERSION_SUFFIX = "${replace(CUDA_VERSION, ".", "-")}" }
  contexts = {
    base_image = "target:${plaintext_image_name(base_image)}_base_efa-installer-${replace(efa_installer_version, ".", "-")}${debug_enabled == 1 ? "-debugsyms" : ""}${ item.mpi4_enabled == 1 ? "-mpi4" : "" }${ item.mpi5_enabled == 1 ? "-mpi5" : ""}"
  }
  matrix = {
    base_image = [
      "amazonlinux:2",
      "amazonlinux:2023",
      "rockylinux:8",
      "rockylinux:9",
      "opensuse/leap:15",
      # Intentionaly not included
      # "centos:centos7"
      "debian:10",
      "debian:11",
      #"debian:12",
      "ubuntu:20.04",
      "ubuntu:22.04",
      "ubuntu:24.04",
    ]
    efa_installer_version = [ "1.34.0", "1.33.0" ]
    cuda_version = [
      "12.6",
      "12.5",
      "12.4",
      "12.3",
      "12.2",
      "12.1",
      "12.0",
      "11.8",
      "11.7",
    ]
  }
}

# Generate a `make dist` tarball. Note that this requires ./configure to be
# called, and that the contents of this "dist tarball" may differ depending on
# the configuration options passed. Requires dependencies to be installed as
# ./configure aborts if they cannot resolve.
#target "makedist" {
#  name = "makedist-${item.accelerator}"
#  matrix = {
#    item = [
#      { accelerator = "neuron", base_image = "target:ubuntu2204_efa-installer-${replace(, ".", "-")}" },
#      { accelerator = "cuda",   base_image = "target:ubuntu2204_efa-installer-${replace(EFA_INSTALLER_VERSION, ".", "-")}" },
#    ]
#  }
#  contexts = { src = ".", base_image = "${item.base_image}" }
#  args = { ACCELERATOR = item.accelerator }
#  dockerfile = ".docker/containers/Dockerfile.makedist"
#  output = ["type=local,dest=dockerbld/tarball"]
#}

# # Generate a universal srpm using packit.
# target "srpm" {
#   contexts = { src = ".", makedist = "target:makedist-neuron" }
#   dockerfile = ".docker/containers/Dockerfile.srpm"
#   output = ["type=local,dest=dockerbld/srpm"]
# }
#
# # Generate RPMs from the srpm above.
# target "rpms" {
#   name = "pkg${item.aws == "1" ? "-aws" : ""}-${replace(item.family, "/", "_")}-${replace(item.version, ".", "_")}"
#   matrix = {
#     item = [
#       {
#         family = "amazonlinux",
#         package_frontend = "dnf",
#         version = "2023",
#         efa = "latest",
#         cuda_distro = "amzn2023",
#         toolkit_version = "12-6",
#         accelerator = "cuda",
#         enable_powertools = "0",
#         aws = "1"
#       },
#       {
#         family = "amazonlinux",
#         package_frontend = "yum",
#         version = "2",
#         efa = "latest",
#         cuda_distro = "rhel7",
#         toolkit_version = "12-3",
#         accelerator = "cuda",
#         enable_powertools = "0",
#         aws = "1"
#       },
#       {
#         family = "rockylinux",
#         package_frontend = "dnf",
#         version = "8",
#         efa = "latest",
#         cuda_distro = "rhel8",
#         toolkit_version = "12-6",
#         accelerator = "cuda",
#         enable_powertools = "1",
#         aws = "1"
#       },
#       {
#         family = "rockylinux",
#         package_frontend = "dnf",
#         version = "9",
#         efa = "latest",
#         cuda_distro = "rhel9",
#         toolkit_version = "12-6",
#         accelerator = "cuda",
#         enable_powertools = "0",
#         aws = "1"
#       },
#     ]
#   }
#   contexts = {
#     efainstaller = "target:efainstaller"
#     srpm = "target:srpm"
#   }
#   dockerfile = ".docker/containers/Dockerfile.${item.package_frontend}"
#   output = ["type=local,dest=dockerbld/pkgs"]
#   args = {
#     FAMILY = item.family,
#     VERSION = item.version
#     EFA_INSTALLER_VERSION = item.efa
#     CUDA_DISTRO = item.cuda_distro
#     VARIANT = item.accelerator
#     AWS_BUILD = item.aws
#     TOOLKIT_VERSION = item.toolkit_version
#     ENABLE_POWERTOOLS = item.enable_powertools
#   }
# }
#
# # Build and package for debian-like distributions by building and invoking fpm.
# target "debs" {
#   name = "pkg-${item.accelerator}${item.aws == "1" ? "-aws" : ""}-${replace(item.family, "/", "_")}-${replace(item.version, ".", "_")}"
#   matrix = {
#     item = [
#        { accelerator = "cuda", aws = "1", family = "debian", version = "oldstable", cuda_distro = "debian11" },
#        # XXX: EFA Installer lacks support.
#        #{ accelerator = "cuda", aws = "1", platform = "amd64", family = "debian", version = "stable", cuda_distro = "debian11" },
#        { accelerator = "cuda", aws = "1", family = "ubuntu", version = "20.04",     cuda_distro = "ubuntu2004" },
#        { accelerator = "cuda", aws = "1", family = "ubuntu", version = "22.04",     cuda_distro = "ubuntu2204" },
#        { accelerator = "cuda", aws = "1", family = "ubuntu", version = "24.04",     cuda_distro = "ubuntu2404" },
#        { accelerator = "cuda", aws = "0", family = "debian", version = "oldstable", cuda_distro = "debian11" },
#        # XXX: EFA Installer lacks support.
#        #{ accelerator = "cuda", aws = "0", family = "debian", version = "stable", cuda_distro = "debian11" },
#        { accelerator = "cuda", aws = "0", family = "ubuntu", version = "20.04",     cuda_distro = "ubuntu2004" },
#        { accelerator = "cuda", aws = "0", family = "ubuntu", version = "22.04",     cuda_distro = "ubuntu2204" },
#        { accelerator = "cuda", aws = "0", family = "ubuntu", version = "24.04",     cuda_distro = "ubuntu2404" },
#
#        # XXX: todo
#        # { accelerator = "neuron", aws = "1", platform = "amd64", family = "debian", version = "oldstable", cuda_distro = "debian11" },
#        # #{ accelerator = "neuron", aws = "1", platform = "amd64", family = "debian", version = "stable", cuda_distro = "debian11" },
#        # { accelerator = "neuron", aws = "1", platform = "amd64", family = "ubuntu", version = "20.04",     cuda_distro = "ubuntu2004" },
#        # { accelerator = "neuron", aws = "1", platform = "amd64", family = "ubuntu", version = "22.04",     cuda_distro = "ubuntu2204" },
#        # { accelerator = "neuron", aws = "1", platform = "amd64", family = "ubuntu", version = "24.04",     cuda_distro = "ubuntu2404" },
#
#        # { accelerator = "neuron", aws = "0", platform = "amd64", family = "debian", version = "oldstable", cuda_distro = "debian11" },
#        # #{ accelerator = "neuron", aws = "0", platform = "amd64", family = "debian", version = "stable", cuda_distro = "debian11" },
#        # { accelerator = "neuron", aws = "0", platform = "amd64", family = "ubuntu", version = "20.04",     cuda_distro = "ubuntu2004" },
#        # { accelerator = "neuron", aws = "0", platform = "amd64", family = "ubuntu", version = "22.04",     cuda_distro = "ubuntu2204" },
#        # { accelerator = "neuron", aws = "0", platform = "amd64", family = "ubuntu", version = "24.04",     cuda_distro = "ubuntu2404" },
#     ]
#   }
#   contexts = {
#     efainstaller = "target:efainstaller"
#     makedist = "target:makedist-${item.accelerator}"
#   }
#   dockerfile = ".docker/containers/Dockerfile.dpkg"
#   output = ["type=local,dest=dockerbld/pkgs"]
#   args = {
#     FAMILY = item.family,
#     VERSION = item.version
#     CUDA_DISTRO = item.cuda_distro
#     AWS_BUILD = item.aws
#   }
# }
#
# target "nccl_tests" {
#   matrix = {
#
#   }
#   contexts = {
#     nccl_tests = "https://github.com/NVIDIA/nccl-tests.git"
#   }
# }
#
