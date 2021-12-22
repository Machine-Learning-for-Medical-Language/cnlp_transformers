# Contributing

## For Maintainers: Building and uploading new package version

### Building

First, increment the version number in `src/cnlpt/VERSION`.

Then, make sure PyPA `build` is installed and run it:

```sh
$ pip install --upgrade build
$ python -m build
```

This will build the package in the `./dist/` directory, creating it if it does not exist.

### Uploading

First, set up your PyPI API key:

0. Log into your PyPI account
0. Generate an API key for the `cnlp-transformers` project [here](https://pypi.org/manage/account/#api-tokens)
0. Create a file `~/.pypirc`:
```cfg
[pypi]
username = __token__
password = <the token value, including the `pypi-` prefix>
```

Next, make sure PyPA `twine` is installed:

```sh
$ pip install --upgrade twine
```

Then, upload to PyPI:

```sh
$ python -m twine upload dist/*
```
