{
  description = "ContentAPI Discord Bridge";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in
  {
    devShells.${system}.default =
      pkgs.mkShell
        {
          buildInputs = with pkgs; [
            python312
	          python312Packages.flask
	          python312Packages.flask-cors
	          python312Packages.flask-sock
	          python312Packages.websockets
	          python312Packages.torch
	          python312Packages.transformers
	          ruff
          ];
        };
  };
}
