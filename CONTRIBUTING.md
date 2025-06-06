# Contributing

## Developing `cnlp-transformers`

To contribute to the development of cnlp-transformers, please follow these steps.

### Fork the repository

1. Fork this project on GitHub. (Click "Fork" near the top of the project homepage.)

   Leave the repository name the same, and select "Copy the default branch only".

2. Clone your fork to your local machine.

   ```bash
   git clone https://github.com/{your username}/cnlp_transformers.git
   cd cnlp_transformers
   ```

3. Add this repository as your upstream remote.

   ```bash
   git remote add upstream https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers.git
   ```

   Now running `git remote -v` should show:

   ```txt
   origin  https://github.com/{your username}/cnlp_transformers.git (fetch)
   origin  https://github.com/{your username}/cnlp_transformers.git (push)
   upstream        https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers.git (fetch)
   upstream        https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers.git (push)
   ```

### Set up your Python environment

You can set a python development environment using a number of tools,
we have instructions for using [uv](https://github.com/astral-sh/uv)
(recommended) or conda.

#### Using uv (recommended)

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/).

2. From the project's base directory, run:

   ```bash
   # uv will use python 3.13 and install dev dependencies by default.
   # If you prefer a different python version, use e.g. `uv sync -p 3.9`
   uv sync 
   source .venv/bin/activate # activate the virtual environment
   ```

#### Using conda

1. Install conda or [miniconda](https://docs.anaconda.com/miniconda/).

2. Create a new conda environment:

   ```bash
   conda create -n cnlpt python=3.11 # 3.9 and 3.10 are also supported
   conda activate cnlpt
   ```

3. From the project's base directory, install dependencies:

   ```bash
   # editable install with dev dependencies
   pip install dependency-groups
   dependency-groups dev | xargs pip install -e.
   ```

### Development tools

#### Pre-commit hooks

   Install the pre-commit hooks with:

   ```sh
   make hooks
   ```

   This will automatically double check the code style before you make any
   commit, and warn you if there are any linting or formatting errors.

#### Linting and formatting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

Run `make check` to lint and format your code.

If you use VSCode, there is a [ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
that might be handy for development (e.g., format on save).

#### Testing your code

You can run the pytest test suite with `make test`.

#### CI

This repository has GitHub Actions set up to automatically ensure the
codebase is linted and formatted and that the test suite in `test/` is
passing.

These actions will run whenever a commit is pushed or a pull request is
opened that makes changes to any of the following files:

- `src/**`
- `test/**`
- `pyproject.toml`

The `lint-and-format` workflow should always pass if `make check` reports
that everything is correct.

The `build-and-test` workflow will run `pytest` on Linux, MacOS, and Windows,
for each Python version this project supports (currently 3.9, 3.10, and 3.11).

You can see the structure of these CI runs in the
[**Actions**](https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers/actions)
tab of this repository.

### Proposing changes

If you have changes to the code that you wish to contribute to the
repository, please follow these steps.

1. Create and checkout a new branch on your fork for the changes. For example:

   ```bash
   git checkout -b my-new-feature
   ```

2. Start developing! Commit your changes to your new branch.

3. When you're ready, run `make check` and `make test` to make sure the
   linter and formatter like your code, and that the test suite passes.

4. When this is done and all your changes are pushed to your fork,
   you can open a pull request for us to review your changes.
   Link any related issues that your PR addresses.

## Instructions for maintainers

### Updating the changelog

All new features and changes should be added to [`CHANGELOG.md`](CHANGELOG.md)
under the "Unreleased" heading. A new heading will be added on every release to
absorb all the unreleased changes.

### Releasing the next version

#### 1. Choosing a version number

When deciding whether to create a major, minor, or patch version, follow
the Semantic Versioning guidelines. The key points are as follows:

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1. MAJOR version when you make incompatible API changes
> 2. MINOR version when you add functionality in a backwards compatible manner
> 3. PATCH version when you make backwards compatible bug fixes

> [!NOTE]
> At time of writing we are still in major version 0, meaning we are
> in the initial development stage. For major version 0, both incompatible
> API changes and backwards compatible feature adds fall under the MINOR
> version, and the API is not considered stable.

#### 2. Creating a PR for the release

When the codebase is ready for release, run `python scripts/prepare_release.py <new version number>`.
This will walk you through the last few changes that need to be made before release,
including updating the changelog and the lockfile.

> [!WARNING]
> `prepare_release.py` requires uv to update the lockfile.
> It will not work if uv is not installed on your machine.

Once you're done, commit your changes and open a PR on `main` with a title like "Release 0.7.0".
Make sure the CI passes, then you can merge and continue to step 3.

#### 3. Creating a release with GitHub

Go to the [releases](https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers/releases)
page, and click "Draft a new release". Create a new tag with the new version
number preceded by a `v` (e.g. `v0.7.0`), and set the release title to be the same as the tag.
Click "Generate release notes" to automatically generate release notes from the
commit history. You can edit these release notes as necessary. When you're ready,
click "Publish release".

#### 4. Setting up your PyPI API key

1. Log into your PyPI account

2. Generate an API key for the `cnlp-transformers` project
   [here](https://pypi.org/manage/account/#api-tokens)

3. Create a file `~/.pypirc`:

   ```cfg
   [pypi]
   username = __token__
   password = <the token value, including the `pypi-` prefix>
   ```

#### 5. Building and uploading to PyPI

1. Checkout the commit for the new version; this will usually
   be the latest commit in `main`.

2. **Double check that the version number shown by `cnlpt --version`
   has been incremented from the previous version on PyPI.**

3. Build the package using `make build`:

   This will build the package in the `./dist/` directory, creating it if
   it does not exist.

4. Upload to PyPI with `twine`:

    ```sh
    python -m twine upload dist/*
    ```

### Building the documentation

Here are some pointers for updating the Sphinx configuration. This is not exhaustive.

- Whenever a new class from a third party package (usually Transformers) is added
  to a type annotation, a link will need to be added to the Intersphinx mappings.
  For Transformers, you will have to add an entry for every namespace path you use
  in the code; for instance, if you import `InputExample` from `transformers` and
  from `transformers.data.processing.utils`, you will need two lines in `transformer_objects.txt` as follows:
  
  ```txt
  transformers.InputExample py:class 1 main_classes/processors#$ -
  transformers.data.processors.utils.InputExample py:class 1 main_classes/processors#transformers.InputExample -
  ```
  
  The specification for the Intersphinx mappings can be found
  [here](https://sphobjinv.readthedocs.io/en/stable/syntax.html).

  To add mappings for other libraries, first check if an `objects.inv` file is
  published for that project somewhere; then add it to `intersphinx_mappings` in `conf.py`
  per the instructions
  [here](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#confval-intersphinx_mapping).

- To rebuild the autodoc toctrees and the `transformers` Intersphinx
  mappings, run `build_doc_source.sh`.

- ReadTheDocs should automatically begin building documentation for the latest
  version upon the creation of the release in GitHub. To build the docs locally
  for testing documentation changes before uploading to readthedocs, first
  **uncomment lines 36 and 65 on `docs/conf.py`,** then execute the following:

  ```sh
  cd docs
  make html
  ```
  
  This will write the docs to `docs/build/html`; simply open
  `docs/build/html/index.html` in your browser of choice to view the
  built documentation.
