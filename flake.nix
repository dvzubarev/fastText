{
  description = "Fast-text fork";
  inputs = {
    textapp-pkgs.url = "git+ssh://git@tsa04.isa.ru/textapp/textapp-pkgs";
  };

  outputs = { self, textapp-pkgs }:
    let pkgs = import textapp-pkgs.inputs.nixpkgs {
          system = "x86_64-linux";
          overlays = [ textapp-pkgs.overlays.default  ];
        };
    in {

      devShells.x86_64-linux.default =
        pkgs.mkShell {

          buildInputs = [
            pkgs.stdenv
            pkgs.cmake
            pkgs.rapidjson
            pkgs.fast_bpe
            pkgs.ccls
          ];

          shellHook = '''';
        };
    };

}
