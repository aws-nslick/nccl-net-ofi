{
  accelerator = [ "-cuda" "-neuron" ];
  platform = [ "-aws" "" ];
  tracing = [ [ "" ] [ "-nvtx" ] [ "-lttng" ] [ "-nvtx" "-lttng" ] ];
  memory = [ "-valgrind" "" ];
  traceprints = [ "-trace" "" ];
  cpp = [ "-cpp" "" ];
  stdenv = [ (pkgs: pkgs.gcc10Stdenv) (pkgs: pkgs.clangStdenv) (pkgs: pkgs.gcc14Stdenv) ];
}
