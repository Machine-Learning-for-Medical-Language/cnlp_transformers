# remove generated rst files
find docs -maxdepth 1 ! -name 'index.rst' -name '*.rst' -type f -exec rm -f {} +

# remove generated links to HF transformers
rm -f docs/transformer_objects.inv

# rebuild transformers links
uv run sphobjinv convert zlib docs/transformer_objects.txt --quiet

# generate rst files
SPHINX_APIDOC_OPTIONS=members,show-inheritance uv run sphinx-apidoc -feTM -o docs src/cnlpt
echo "   :noindex:" >> docs/cnlpt.rst

# build docs as html
uv run sphinx-build -T -b html docs $1/html
