{ pkgs ? (import <textapp-pkgs> {})
, version ? "dev"
}:
let der = import ./default.nix { inherit pkgs version; };
in
pkgs.mkShell {
  name = "fast-text-${version}";

  src = null;

  inputsFrom = [ der ];


  direnvHook =
  ''
    echo "%compile_commands.json" > .ccls
    stdlibpath=${pkgs.stdenv.cc.cc.outPath}/include/c++/${pkgs.stdenv.cc.cc.version}
    echo "-isystem" >> .ccls
    echo "$stdlibpath" >> .ccls
    echo "-isystem" >> .ccls
    echo "$stdlibpath/x86_64-unknown-linux-gnu" >> .ccls
    echo "-isystem" >> .ccls
    echo ${pkgs.stdenv.cc.libc_dev.outPath}/include >> .ccls

    tr -s ' ' '\n' <<< "$NIX_CFLAGS_COMPILE" >> .ccls
  '';

}
