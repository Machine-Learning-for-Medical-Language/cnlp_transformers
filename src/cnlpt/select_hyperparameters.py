#!/usr/bin/env python
import json
from pathlib import Path
from typing import Optional, Tuple

import click
import itertools


def get_job_from_hyperparameters(hyperparameters, indices):
    sorted_pairs = sorted(hyperparameters.items())

    paired_lists = [[(key, value) for value in values] for key, values in sorted_pairs]

    if len(indices) == 1:
        index = indices[0]
        cartesian_product = [dict(job) for job in itertools.product(*paired_lists)]
        return cartesian_product[index]
    elif len(indices) == len(paired_lists):
        return dict(pair[index] for pair, index in zip(paired_lists, indices))
    else:
        raise ValueError(
            f"Expected {len(paired_lists)} indices but received {len(indices)}"
        )


@click.command()
@click.argument(
    "hyperparameters", type=click.Path(path_type=Path, exists=True, dir_okay=False)
)
@click.argument("indices", type=int, nargs=-1, required=True)
@click.option(
    "--default-args",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="If supplied, merges the default arguments found in the provided "
    "file with the indexed experiment arguments. Only necessary if "
    "using different default arguments from what is in cnn/train.py.",
)
def cli(hyperparameters: Path, indices: Tuple[int], default_args: Optional[Path]):
    """
    Produces the hyperparameter dictionary for a job index from a
    hyperparameter search space.

    When providing a job index, either supply a single number or supply
    a space-separated list of numbers indexing into the list of options
    for each hyperparameter, in alphabetical order.
    """
    with hyperparameters.open("r", encoding="utf8") as hp_file:
        hp_dict = json.load(hp_file)

    if default_args is not None:
        with open("default_args.json", "r", encoding="utf8") as def_args_file:
            default_args_dict = json.load(def_args_file)
        hyperparameters = {
            **default_args_dict,
            **get_job_from_hyperparameters(hp_dict, indices),
        }
    else:
        hyperparameters = get_job_from_hyperparameters(hp_dict, indices)

    click.echo(json.dumps(hyperparameters, indent=2))


if __name__ == "__main__":
    cli()
