{ inputs, lib }:
rec {
  mkGitVersion = i: "git${if (i ? rev) then (builtins.substring 0 7 "${i.rev}") else "dirty" + (builtins.substring 0 7 "${i.dirtyRev}")}";
  compilerName = s: "-${if s.cc.isGNU then "gcc" else "clang"}${s.cc.version}";
  genComboName = pkgs: combo: prevname: "${prevname}${(lib.strings.concatStrings combo.tracing)}${combo.memory}${combo.traceprints}${combo.cpp}${compilerName (combo.stdenv pkgs)}";
  genPkgFromCombo = pkgs: combo: (pkgs.callPackage ./default.nix {
      stdenv = (combo.stdenv pkgs);
      cudaSupport = (combo.accelerator == "-cuda");
      neuronSupport = (combo.accelerator == "-neuron");
      enableAwsTuning = (combo.platform == "-aws");
      enableNVTXTracing = (combo.accelerator == "-cuda" && (builtins.elem "-nvtx" combo.tracing));
      enableLTTNGTracing = (builtins.elem "-lttng" combo.tracing);
      enableValgrind = (combo.memory == "-valgrind");
      enableTracePrints = (combo.traceprints == "-trace");
      enableCPPMode = (combo.cpp == "-cpp");
    }).overrideAttrs (pprev: {
      src = inputs.self;
      version = mkGitVersion inputs.self;
    });

  genAttrsFromCombo = { pkgs }: combo: let
    value = (genPkgFromCombo pkgs combo);
    name = (genComboName pkgs combo value.pname);
  in { inherit name; value = value.overrideAttrs { inherit name; }; };
}
