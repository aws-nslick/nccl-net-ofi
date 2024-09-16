{ inputs }:
(final: prev:
let
  lib = prev.lib // (import ./lib.nix { inherit inputs; lib = prev.lib; });
in
{

  libgdrcopy = final.cudaPackages.backendStdenv.mkDerivation rec {
    pname = "libgdrcopy";
    src = inputs.gdrcopy;
    version = lib.mkGitVersion inputs.gdrcopy;
    makeFlags = [
      "LIB_MAJOR_VER=2"
      "LIB_MINOR_VER=5"
      "DESTLIB=$out/lib"
      "DESTINC=$out/include"
      "GDRAPI_ARCH=X86"
    ];
    patchPhase = "chmod +x config_arch";
    buildPhase = "make -C src all";
    depsTargetTarget = with final.cudaPackages; [ cuda_cudart ];
    installPhase = "mkdir -p $out/lib && make ${lib.strings.concatStringsSep " " makeFlags} lib_install";
  };

  rdma-core = prev.rdma-core.overrideAttrs (pprev: {
    src = inputs.rdma-core;
    version = lib.mkGitVersion inputs.rdma-core;
  });

  hwloc = prev.hwloc.overrideAttrs (pprev: {
    src = inputs.hwloc;
    version = lib.mkGitVersion inputs.hwloc;
    nativeBuildInputs = (pprev.nativeBuildInputs or [ ]) ++ [ prev.autoreconfHook ];
  });

  # pmix/prrte/openmpi cannot support new hwloc
  pmix = prev.pmix.override { hwloc = final.hwloc; };
  prrte = prev.prrte.override { hwloc = final.hwloc; };
  openmpi = (prev.openmpi.override {
    cudaSupport = true;
    libfabric = final.libfabric;
    rdma-core = final.rdma-core;
    hwloc = final.hwloc;
  }).overrideAttrs (pprev: {
    src = inputs.openmpi;
    version = lib.mkGitVersion inputs.openmpi;
    nativeBuildInputs = (pprev.nativeBuildInputs or [ ]) ++ [
      prev.autoconf
      prev.automake
      prev.libtool
      prev.perl
      prev.git
      prev.flex
    ];
    prePatch = ''
      patchShebangs .
      ./autogen.pl
    '';
    outputs = final.lib.lists.remove "man" pprev.outputs;
    NIX_CFLAGS_COMPILE = "-Wno-deprecated-declarations";
  });

  libfabric = (prev.libfabric.override {
    enableOpx = false;
    enablePsm2 = false;
  }).overrideAttrs (pprev: {
    src = inputs.libfabric;
    version = lib.mkGitVersion inputs.libfabric;
    configureFlags = (prev.configureFlags or [ ]) ++ [
      "--enable-efa=yes"
      "--with-cuda=${prev.lib.getDev final.cudaPackages.cudatoolkit}"
      "--enable-cuda-dlopen"
      "--with-gdrcopy=${prev.lib.getDev final.libgdrcopy}"
      "--enable-gdrcopy-dlopen"
    ];
    buildInputs = (pprev.buildInputs or [ ]) ++ [
      final.rdma-core
    ];
  });

  cudaPackages = prev.cudaPackages.overrideScope (ffinal: pprev: rec {
    nccl = pprev.nccl.overrideAttrs {
      src = inputs.nccl;
      version = lib.mkGitVersion inputs.nccl;
    };
    nccl-tests = (pprev.nccl-tests.overrideAttrs {
      src = inputs.nccl-tests;
      version = lib.mkGitVersion inputs.nccl-tests;
    }).override {
      mpiSupport = true;
      mpi = final.openmpi;
      cudaPackages = pprev.cudaPackages // { inherit nccl; };
      config.cudaSupport = true;
    };
  });
})
