{ lib
, fetchFromGitHub
, symlinkJoin
, gitUpdater
, stdenv
, config
, libfabric
, hwloc
, autoreconfHook
, lttng-ust
, valgrind
, mpi
, cudaPackages ? { }
, enableTests ? true
, enableTracePrints ? (enableTests)
, neuronSupport ? (!config.cudaSupport)
, cudaSupport ? (config.cudaSupport && !neuronSupport)
, enableLTTNGTracing ? false
, enableNVTXTracing ? false
, enableValgrind ? false
, enableAwsTuning ? false
, enableCPPMode ? false
}:

assert neuronSupport != cudaSupport;
#assert !enableNVTXTracing || (enableNVTXTracing && !neuronSupport);

let
  basename = "lib${if neuronSupport then "nccom" else "nccl"}-net-ofi";
  pname = "${basename}${if enableAwsTuning == true then "-aws" else ""}";
  version = "1.11.0";
  src = fetchFromGitHub {
    owner = "aws";
    repo = "aws-ofi-nccl";
    rev = "v${version}-aws";
    sha256 = "sha256-y3yVPqak+36UXI6L/ddQIfBBwpeiciW571noc8LNefU=";
  };
  cuda_build_deps_joined = symlinkJoin {
    name = "cuda-build-deps-joined";
    paths = lib.optionals (cudaSupport) [
      (lib.getOutput "static" cudaPackages.cuda_cudart)
      (lib.getDev cudaPackages.cuda_cudart)
      (lib.getDev cudaPackages.cuda_nvcc)
    ];
  };
in
stdenv.mkDerivation {
  inherit pname version src;

  enableParallelBuilding = true;
  separateDebugInfo = true;
  strictDeps = true;

  nativeBuildInputs = [ autoreconfHook ];
  configureFlags =
    [
      "--enable-picky-compiler"
      "--enable-werror"
      "--with-hwloc=${lib.getDev hwloc}"
      "--with-libfabric=${lib.getDev libfabric}"
    ]
    ++ lib.optionals enableCPPMode [
      "--enable-cpp=yes"
    ]
    ++ lib.optionals (!enableTests) [
      "--disable-tests"
    ]
    ++ lib.optionals enableTests [
      "--enable-tests"
      "--with-mpi=${lib.getDev mpi}"
    ]
    ++ lib.optionals enableTracePrints [
      "--enable-trace"
    ]
    ++ lib.optionals cudaSupport [
      "--with-cuda=${cuda_build_deps_joined}"
    ]
    ++ lib.optionals enableLTTNGTracing [
      "--with-lttng=${lib.getDev lttng-ust}"
    ]
    ++ lib.optionals enableValgrind [
      "--with-valgrind=${lib.getDev valgrind}"
    ]
    ++ lib.optionals (enableNVTXTracing && cudaSupport) [
      "--with-nvtx=${lib.getDev cudaPackages.cuda_nvtx}"
    ]
    ++ lib.optionals enableAwsTuning [
      "--enable-platform-aws"
    ]
    ++ lib.optionals neuronSupport [
      "--enable-neuron"
    ];

  buildInputs =
    [
      libfabric
      hwloc
    ]
    ++ lib.optionals cudaSupport [
      cuda_build_deps_joined
    ]
    ++ lib.optionals enableValgrind [
      valgrind
    ]
    ++ lib.optionals enableTests [
      mpi
    ]
    ++ lib.optionals enableLTTNGTracing [
      lttng-ust
    ];
  postInstall = ''find $out/lib | grep -E \.la$ | xargs rm'';

  doCheck = enableTests;
  checkPhase = ''
    set -euo pipefail
    for test in $(find tests/unit/ -type f -executable -print | xargs) ; do
      echo "======================================================================"
      echo "Running $test"
      ./$test
      test $? -eq 0 && (echo "✅ Passed" || (echo "❌ Failed!" && exit 1))
    done
    echo "All unit tests passed successfully."
    set +u
  '';

  passthru = {
    inherit cudaSupport;
    updateScript = gitUpdater {
      inherit pname version;
      rev-prefix = "v";
    };
  };
  meta = with lib; {
    homepage = "https://github.com/aws/aws-ofi-nccl";
    license = licenses.asl20;
    broken = (cudaSupport && !config.cudaSupport);
    maintainers = with maintainers; [ sielicki ];
    platforms = [
      "x86_64-linux"
      "aarch64-linux"
    ];
  };
}
