# Contributing

## Developing `cnlp-transformers`

The first things to do when contributing to the code base here are to clone this repository and set up your Python environment.

1. Clone this repository:
```sh
# Either the HTTPS method...
$ git clone https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers.git
# ...or the SSH method
$ git clone git@github.com:Machine-Learning-for-Medical-Language/cnlp_transformers.git
```
2. Enter the repo: `cd cnlp_transformers`
3. You will need Python 3.8. Either a `venv` virtual environment or a Conda environment should work. Create your environment and activate it.
4. Install the development dependencies: `pip install -r dev-requirements.txt`
5. Install `cnlp-transformers` in editable mode: `pip install -e .`

**The remainder of the instructions on this document will assume that you have installed the development dependencies.**

### Proposing changes

If you have changes to the code that you wish to contribute to the repository, please follow these steps.

1. Fork the project on GitHub.
2. Add your fork as a remote to push your work to. Replace
    `{username}` with your username. This names the remote "fork", the
    default Machine-Learning-for-Medical-Language remote is "origin".
```sh
# Either the HTTPS method...
$ git remote add fork https://github.com/{username}/cnlp_transformers.git
# ...or the SSH method
$ git remote add fork git@github.com:{username}/cnlp_transformers.git
```
3. Make a new branch and set your fork as the upstream remote:
```sh
$ git switch -c your-branch-name  # or git checkout -b
$ git push --set-upstream fork your-branch-name
```
4. Open an issue that motivates the change you are making if there is not one already.
5. Make your changes in `your-branch-name` on your fork.
6. Open a PR to close the issue.

## For Maintainers: Building and uploading new package version

### Building

First, increment the version number in `src/cnlpt/VERSION`.

Then, run `build`:

```sh
$ python -m build
```

This will build the package in the `./dist/` directory, creating it if it does not exist.

### Uploading

First, set up your PyPI API key:

1. Log into your PyPI account
1. Generate an API key for the `cnlp-transformers` project [here](https://pypi.org/manage/account/#api-tokens)
1. Create a file `~/.pypirc`:
```cfg
[pypi]
username = __token__
password = <the token value, including the `pypi-` prefix>
```

Then, upload to PyPI with `twine`:

```sh
$ python -m twine upload dist/*
```
