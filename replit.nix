{ pkgs }: 

let
  python = pkgs.python38;
in
{
  buildInputs = [
    pkgs.gcc
    pkgs.libffi
    pkgs.zlib
    pkgs.openssl
    python
    pkgs.python38Packages.pytorch
    pkgs.python38Packages.spacy
    pkgs.python38Packages.transformers
    pkgs.python38Packages.pinecone
    pkgs.python38Packages.mailbox
  ];

  SHELL = "${python}/bin/python";
}
