#!/usr/bin/env bash
find docs -maxdepth 2 ! -name 'index.rst' -name '*.rst' -type f -exec rm -f {} +
rm -rf docs/build
rm -f transformer_objects.inv
yes | sphobjinv convert zlib docs/source/transformer_objects.txt &&
sphinx-apidoc -feT -o docs/source src/cnlpt &&
cd docs &&
O=-a make html
