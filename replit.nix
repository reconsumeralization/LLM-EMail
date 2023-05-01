{ pkgs }: 

let
  python = pkgs.python38;
in
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

nix-build email_agent.nix

all:
    make email_agent
    make agent_manager

email_agent:
    nix-build email_agent.nix

agent_manager:
    nix-build agent_manager.nix
