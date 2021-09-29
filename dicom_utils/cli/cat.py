#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
from argparse import ArgumentParser

import pydicom

from ..metadata import add_patient_age, dicom_to_json, drop_empty_tags, drop_fields_by_length


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("file", help="DICOM file to print")
    parser.add_argument("--output", "-o", help="output format", choices=["txt", "json"], default="txt")
    parser.add_argument(
        "--max_length", "-l", help="drop fields with values longer than MAX_LENGTH", default=100, type=int
    )
    parser.add_argument("--tags", "-t", nargs="+", help="specific tags", default=None)
    return parser


def main(args: argparse.Namespace) -> None:
    with pydicom.dcmread(args.file, specific_tags=args.tags, stop_before_pixels=True) as dcm:
        dcm = drop_empty_tags(dcm)
        add_patient_age(dcm)
        drop_fields_by_length(dcm, args.max_length, inplace=True)

        if args.output == "txt":
            print(f"Metadata for {args.file}")
            if args.tags:
                for v in dcm.values():
                    print(v)
            else:
                print(dcm)
        elif args.output == "json":
            print(json.dumps(dicom_to_json(dcm), indent=2))


def entrypoint():
    parser = get_parser()
    try:
        main(parser.parse_args())
    except AttributeError as e:
        if "DirectoryRecordSequence" in str(e):
            return
        raise e


if __name__ == "__main__":
    entrypoint()
