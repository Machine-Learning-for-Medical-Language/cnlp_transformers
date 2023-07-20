#!/usr/bin/env bash
find docs -maxdepth 1 ! -name 'index.rst' -name '*.rst' -type f -exec rm -f {} +
rm -f transformer_objects.inv
yes | sphobjinv convert zlib docs/transformer_objects.txt &&
SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc -feT -o docs src/cnlpt &&
echo "   :noindex:" >> docs/cnlpt.rst
