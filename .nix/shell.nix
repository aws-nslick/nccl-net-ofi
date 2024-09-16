{ self, system, inputs, pkgs }:
pkgs.mkShell {
    inputsFrom = [ self.packages.${system}.default ];
    packages = [
        pkgs.clang-analyzer
        pkgs.clang-tools
        pkgs.clang
        pkgs.gcc
        pkgs.gdb

        pkgs.ccache
        pkgs.cppcheck
        pkgs.universal-ctags
        pkgs.rtags
        pkgs.ripgrep
        pkgs.irony-server
        pkgs.bear
        pkgs.compdb
        pkgs.act
        pkgs.actionlint

        pkgs.gh
        pkgs.git
        pkgs.eksctl
        pkgs.awscli2

        pkgs.nixpkgs-fmt
  ];
}
