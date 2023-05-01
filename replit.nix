{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "myenv";
  buildInputs = with pkgs.python39Packages; [
    beautifulsoup4
    colorama
    distro
    duckduckgo-search
    gTTS
    google-api-python-client
    openai
    orjson
    pinecone-client
    playsound
    Pillow
    pyyaml
    readability-lxml
    redis
    requests
    selenium
    spacy
    spacy_models.en_core_web_sm
    tiktoken
    tweepy
    webdriver-manager
    click
    jsonschema
    pytest
    pytest-asyncio
    pytest-benchmark
    pytest-cov
    pytest-integration
    pytest-mock
    vcrpy
    pytest-vcr
  ];
}
