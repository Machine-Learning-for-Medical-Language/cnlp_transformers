#!/usr/bin/env bash
yes | sphobjinv convert zlib docs/transformer_objects.txt &&
sphinx-apidoc -f -o docs src/cnlpt &&
sphinx-build -a -b html docs/source docs/build
