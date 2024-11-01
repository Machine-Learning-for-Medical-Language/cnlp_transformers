#!/usr/bin/env python3

import argparse
import os
import subprocess

# At time of writing this comment, the cnn and hier models see to be works in progress, so aren't included here.
MODELS = [
    "current",
    "dtr",
    "event",
    "negation",
    "temporal",
    "termexists",
    "timex",
]

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="append", choices=["all"] + MODELS)
parser.add_argument("--processor", choices=["all", "cpu", "gpu"], default="all")
parser.add_argument("--push", action="store_true", default=False)
args = parser.parse_args()


def get_latest_pip_version(package: str) -> str:
    """Query pip for the latest release of a software package"""
    process = subprocess.run(
        # Use a python version that matches the Dockerfiles
        ["pip", "index", "--python-version=3.9", "versions", package],
        capture_output=True,
        check=True,
    )
    last_line = process.stdout.decode("utf8").strip().split("\n")[-1].strip()
    if "LATEST:" not in last_line:
        raise SystemExit("Did not understand 'pip index versions' output")
    return last_line.split()[-1]


def build_one(model: str, processor: str, *, version: str, push: bool = False) -> None:
    """Builds a single docker image"""
    print(f"Building model {model} for processor {processor}:")

    pwd = os.path.dirname(__file__)

    version_parts = version.split(".")
    major = version_parts[0]
    minor = version_parts[1]
    patch = version_parts[2]

    platforms = "linux/amd64"
    if processor == "cpu" and push:  # only build extra platforms on push because --load can't do multi-platforms
        platforms += ",linux/arm64"

    build_args = [
        f"--build-arg=cnlpt_version={version}",  # to make sure that we don't have a version mismatch, we pin cnlpt
        f"--file={pwd}/Dockerfile.{processor}",
        f"--platform={platforms}",
        f"--tag=smartonfhir/cnlp-transformers:{model}-latest-{processor}",
        f"--tag=smartonfhir/cnlp-transformers:{model}-{major}-{processor}",
        f"--tag=smartonfhir/cnlp-transformers:{model}-{major}.{minor}-{processor}",
        f"--tag=smartonfhir/cnlp-transformers:{model}-{major}.{minor}.{patch}-{processor}",
        f"--target={model}",
        pwd,
    ]
    if push:
        build_args.append("--push")  # to push to docker hub
    else:
        build_args.append("--load")  # to load into docker locally

    subprocess.run(["docker", "buildx", "build"] + build_args, check=True)


if __name__ == '__main__':
    if args.processor == "all":
        processors = ["cpu", "gpu"]
    else:
        processors = [args.processor]

    models = args.model
    if not args.model or "all" in args.model:
        models = MODELS

    # Check version of cnlpt available via pip.
    # Our Dockerfiles pull directly from pip, so we want to be setting the same version as we'll install.
    # We don't want to pull the version from our sibling code in this repo, because it might not be released yet,
    # but we still want to be able to push new builds of the existing releases.
    version = get_latest_pip_version("cnlp-transformers")

    for model in models:
        for processor in processors:
            build_one(model, processor, version=version, push=args.push)
