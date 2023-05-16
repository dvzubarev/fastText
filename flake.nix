{
  description = "Fast-text fork";
  inputs = {
    textapp-pkgs.url = "git+ssh://git@tsa04.isa.ru/textapp/textapp-pkgs?ref=flakes";
  };

  outputs = { self, textapp-pkgs }:
    let pkgs = import textapp-pkgs.inputs.nixpkgs {
          system = "x86_64-linux";
          overlays = [ textapp-pkgs.overlay  ];
        };
    in {

      devShells.x86_64-linux.default =
        pkgs.mkShell {

          buildInputs = [
            pkgs.cmake
            pkgs.rapidjson
            pkgs.fast_bpe
            pkgs.ccls
          ];

          shellHook = '''';
        };
    };

}
