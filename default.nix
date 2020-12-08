{
  pkgs ? import <textapp-pkgs> {},
  version ? "dev"
}:
with pkgs;
stdenv.mkDerivation rec {
  name = "fast-text-${version}";
  inherit version;
  src = if lib.inNixShell then null else ./.;

  nativeBuildInputs = [ cmake ];
  buildInputs = [
    rapidjson
    fast_bpe
  ];
}
