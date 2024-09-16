# Copyright (c) 2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
{
  description = "libnccl-net-ofi flake. (UNSUPPORTED; INTENDED FOR DEVELOPERS ONLY)";

  outputs = inputs: with inputs;
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ (import ./.nix/overlay.nix { inherit inputs; }) ];
            config = {
              cudaSupport = true;
              allowUnfree = true;
            };
          };
          lib = pkgs.lib // (import ./.nix/lib.nix { inherit inputs; inherit (pkgs) lib; });
          base-packages = builtins.listToAttrs (lib.attrsets.mapCartesianProduct (lib.genAttrsFromCombo { inherit pkgs; }) (import ./.nix/combos.nix));
          all-dep-packages = (pkgs.linkFarmFromDrvs "all_deps" (with pkgs; [
            cudaPackages.nccl-tests
            cudaPackages.nccl
            openmpi
            libfabric
            rdma-core
            libgdrcopy
            hwloc
          ]));
          filterPackages = s: (lib.attrsets.filterAttrs (k: v: lib.strings.hasInfix s k) base-packages);
          makeCheck = name: filterString: pkgs.linkFarmFromDrvs name (builtins.attrValues (filterPackages filterString));
        in
        {
          inherit _base-packages;
          checks = rec {
            all = (makeCheck "all" "");
            default = all;
            deps = all-dep-packages;
            neuron = (makeCheck "neuron" "neuron");
            cuda = (makeCheck "cuda" "cuda");
            lttng = (makeCheck "lttng" "lttng");
            nvtx = (makeCheck "nvtx" "nvtx");
            clang = (makeCheck "clang" "clang");
            gcc7 = (makeCheck "gcc7" "gcc7");
            gcc14 = (makeCheck "gcc14" "gcc14");
            cpp = (makeCheck "cpp" "cpp");
            valgrind = (makeCheck "valgrind" "valgrind");
            debug = (makeCheck "debug" "debug");
            aws = (makeCheck "aws" "aws");
          };
          packages = {
            inherit (pkgs)
              libfabric
              openmpi
              rdma-core
              libgdrcopy
              hwloc
              ;
            inherit (pkgs.cudaPackages)
              nccl
              nccl-tests
              ;
            default = base-packages."libnccl-net-ofi-aws-lttng-clang18.1.8";
          };
          devShells.default = pkgs.callPackage ./.nix/shell.nix {
            inherit self inputs pkgs system;
          };
        }
      ) // {
      githubActionChecks = inputs.nix-github-actions.lib.mkGithubMatrix {
        checks = nixpkgs.lib.getAttrs [ "x86_64-linux" ] self.outputs.base-packages;
      };
    };

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/DeterminateSystems/nixpkgs-weekly/0.1.678339.tar.gz";
    flake-utils.url = "github:numtide/flake-utils";
    systems.url = "github:nix-systems/default-linux";
    flake-utils.inputs.systems.follows = "systems";
    nix-github-actions.url = "github:nix-community/nix-github-actions";
    nix-github-actions.inputs.nixpkgs.follows = "nixpkgs";

    nccl = { url = "github:NVIDIA/nccl"; flake = false; };
    nccl-tests = { url = "github:NVIDIA/nccl-tests"; flake = false; };
    gdrcopy = { url = "github:NVIDIA/gdrcopy"; flake = false; };
    rdma-core = { url = "github:linux-rdma/rdma-core/master"; flake = false; };
    libfabric = { url = "github:ofiwg/libfabric/main"; flake = false; };
    openmpi = { url = "git+https://github.com/open-mpi/ompi.git?ref=main&submodules=1"; flake = false; };
    hwloc = { url = "github:open-mpi/hwloc/v2.x"; flake = false; };
  };

  nixConfig = {
    allowUnfree = true;
    cudaSupport = true;
    extra-substituters = [
      "https://numtide.cachix.org"
      "https://nix-community.cachix.org"
      "https://devenv.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "numtide.cachix.org-1:2ps1kLBUWjxIneOy1Ik6cQjb41X0iXVXeHigGmycPPE="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };
}
