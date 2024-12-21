import os
import re
import subprocess
import sys

import tomllib


def validate_version_str(version_str: str):
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version_str)
    if match is None:
        print(f"'{version_str}' is not a valid version string; must be 'X.Y.Z'.")
        exit(1)
    return ".".join(match.groups())


def get_fallback_version():
    with open("pyproject.toml", "rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    return pyproject["tool"]["setuptools_scm"]["fallback_version"]


def prepare_release_interactive(version: str):
    version = validate_version_str(version)
    print(f"Preparing to release {version}\n")

    # Update changelog
    print(
        "Update the changelog:\n",
        f"  In CHANGELOG.md, move everything in 'Unreleased' to a new header for {version}.",
        "  Also feel free to add in any missing notable features included in this release.",
        sep="\n",
    )
    input("  Press enter to continue.")
    print()

    # Update setuptools_scm fallback version
    if get_fallback_version() != version:
        print("Update the setuptools_scm fallback version:\n")
    while (current := get_fallback_version()) != version:
        print(
            f'  In pyproject.toml, under [tool.setuptools_scm], change fallback_version to "{version}" (currently "{current}")',
        )
        input("  Press enter to continue.")
    print()

    # Update lockfile
    print(
        "Updating lockfile and venv with `uv sync --reinstall-package cnlp_transformers`..."
    )
    subprocess.run(
        ["uv", "sync", "--reinstall-package", "cnlp_transformers"],
        env=os.environ.copy() | {"SETUPTOOLS_SCM_PRETEND_VERSION": version},
    )
    print(
        f'Done! Before committing your changes, make sure `cnlpt --version` outputs "{version}".'
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Error: please provide a version for the release (e.g., `python prepare_release.py 1.2.3`)`"
        )
        exit(1)
    elif len(sys.argv) > 2:
        print("Error: too many arguments")
        exit(1)
    prepare_release_interactive(sys.argv[1])
