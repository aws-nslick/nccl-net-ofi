#
# Copyright (c) 2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
# What is this? 
#  This overrides defaults for caching, targets, tags, platforms, etc. This
#  file is expected to be symlinked into the root of the tree to select the
#  builder used by `docker buildx bake'.
# 
# This file configures local builds.
# 
# See https://docs.docker.com/build/bake/reference/#file-format

variable "VERSION" { default = "master" }

target "efainstaller" {
  platforms = [ "linux/amd64", "linux/arm64" ]
  context = "."
  dockerfile = ".docker/containers/Dockerfile.efa"
  output = ["type=cacheonly"]
}
