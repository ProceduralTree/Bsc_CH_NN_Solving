{ pkgs ? import <nixpkgs> { } }:
let
  pypackages = ps:
    with ps; [
      tensorflowWithCuda
      numbaWithCuda
      pytorch-bin
      matplotlib
      numpy
      seaborn
      scipy

      # Dev  deps
      python-lsp-server
      python-lsp-ruff
      pylsp-mypy
      python-lsp-black
      black
      ruff-lsp

    ];
in pkgs.mkShell {
  name = "cuda-env-shell";
  buildInputs = with pkgs; [
    git
    gitRepo
    gnupg
    autoconf
    curl
    procps
    gnumake
    util-linux
    m4
    gperf
    unzip
    cudatoolkit
    linuxPackages.nvidia_x11
    libGLU
    libGL
    xorg.libXi
    xorg.libXmu
    freeglut
    xorg.libXext
    xorg.libX11
    xorg.libXv
    xorg.libXrandr
    zlib
    ncurses5
    stdenv.cc
    binutils
    (python3.withPackages pypackages)
  ];
  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
}