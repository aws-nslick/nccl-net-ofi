{
  accelerator = [ "-cuda" "-neuron" ];
  platform = [ "-aws" "" ];
  tracing = [ [ "" ] [ "-nvtx" ] [ "-lttng" ] [ "-nvtx" "-lttng" ] ];
  debug = [ "-debug" "" ];
  memory = [ "-valgrind" "" ];
  traceprints = [ "-trace" "" ];
  cpp = [ "-cpp" "" ];
  stdenv = [ (pkgs: pkgs.gcc7Stdenv) (pkgs: pkgs.clangStdenv) (pkgs: pkgs.gcc14Stdenv) ];
}
