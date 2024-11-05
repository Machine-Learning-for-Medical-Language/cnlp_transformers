.PHONY: help
help:
	@echo 'Targets:'
	@echo '  help   - show this help message'
	@echo '  hooks  - install pre-commit hooks'
	@echo '  check  - lint and format using ruff'
	@echo '  test   - run tests with pytest'
	@echo '  docs   - build the docs'
	@echo '  build  - build cnlp-transformers for distribution'

.PHONY: hooks
hooks:
	pre-commit install

.PHONY: check
check:
	ruff check --fix
	ruff format

.PHONY: test
test:
	pytest test/

.PHONY: docs
docs:
# this script is copied from the old build_doc_source.sh script
	find docs -maxdepth 1 ! -name 'index.rst' -name '*.rst' -type f -exec rm -f {} +
	rm -f transformer_objects.inv
	sphobjinv convert zlib docs/transformer_objects.txt --quiet
	SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc -feT -o docs src/cnlpt
	echo "   :noindex:" >> docs/cnlpt.rst

.PHONY: build
build:
	@if python -m build; then\
		echo 'To upload to PyPI run: `python -m twine upload dist/*`';\
	fi
