# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse as ap
import importlib

from .. import KNOWN_DATASETS


def main():
    # parse args
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description="Prepare a dataset for scene analysis."
    )
    subparsers = parser.add_subparsers(
        help='Dataset to prepare.',
        dest='dataset',
        required=True
    )
    for dataset in KNOWN_DATASETS:
        subparsers.add_parser(dataset, add_help=False)   # redirect help

    parsed_args, remaining_args = parser.parse_known_args()

    # import dataset module
    dataset_module = importlib.import_module(
        name=f'..datasets.{parsed_args.dataset}.prepare_dataset',
        package=__package__
    )

    # run prepare function
    dataset_module.main(remaining_args)


if __name__ == '__main__':
    main()
