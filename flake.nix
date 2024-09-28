{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.05";
    nixpkgs-23-11.url = "github:NixOS/nixpkgs/release-23.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    wgsl-analyzer.url = "github:wgsl-analyzer/wgsl-analyzer";
    wgsl-analyzer.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ (import inputs.rust-overlay) ];
        };

        pkgs-23-11 = import inputs.nixpkgs-23-11 { inherit system; };

        rust = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" ];
        };

        wgsl-analyzer = inputs.wgsl-analyzer.packages.${system}.default;
      in
      {
        devShells.default =
          pkgs.mkShell
            {
              buildInputs = [
                # rust and profiling/debugging
                rust
                wgsl-analyzer
                pkgs.cargo-flamegraph
                pkgs.lld
                pkgs.llvmPackages.bintools
                # older tracy because it's currently broken on 24.05
                pkgs-23-11.tracy
                pkgs.renderdoc
                # wgpu C dependencies
                pkgs.pkg-config
                pkgs.xorg.libX11
                # misc
                pkgs.just
              ];
              # make C libraries available
              LD_LIBRARY_PATH = with pkgs.xorg; with pkgs.lib.strings;
                concatStrings (intersperse ":" [
                  "${libX11}/lib"
                  "${libXcursor}/lib"
                  "${pkgs.libxkbcommon}/lib"
                  "${libXxf86vm}/lib"
                  "${libXi}/lib"
                  "${libXrandr}/lib"
                  "${pkgs.vulkan-loader}/lib"
                  "${pkgs.stdenv.cc.cc.lib}/lib64"
                ]);
            };
      });
}
