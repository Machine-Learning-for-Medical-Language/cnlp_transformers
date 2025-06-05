.PHONY: help
help:
	@echo 'Targets:'
	@echo '  help   - show this help message'
	@echo '  hooks  - install pre-commit hooks'
	@echo '  check  - lint and format using ruff'
	@echo '  test   - run tests with pytest'
	@echo '  docs   - build the docs locally'
	@echo '  build  - build cnlp-transformers for distribution'

.PHONY: hooks
hooks:
	pre-commit install

.PHONY: check
check:
	ruff check --fix
	ruff format
	pre-commit run -a

.PHONY: test
test:
	pytest test/

.PHONY: docs
docs:
	scripts/build_html_docs.sh docs/build
	@echo "Point your browser at file://${PWD}/docs/build/html/index.html to view."

.PHONY: build
build:
	@printf "Are you sure? This will remove everything currently in dist/ and create a new build. [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -f dist/*
	@if python -m build; then\
		echo 'To upload to PyPI run: `python -m twine upload dist/*`';\
	fi
