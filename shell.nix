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
    tr -s ' ' '\n' <<< "$NIX_CFLAGS_COMPILE" >> .ccls
  '';

}
