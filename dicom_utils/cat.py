#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
from argparse import ArgumentParser

import pydicom

from .metadata import add_patient_age, dicom_to_json, drop_empty_tags, drop_fields_by_length


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("file", help="DICOM file to print")
    parser.add_argument("--output", "-o", help="output format", choices=["txt", "json"], default="txt")
    parser.add_argument(
        "--max_length", "-l", help="drop fields with values longer than MAX_LENGTH", default=100, type=int
    )
    return parser


def main(args: argparse.Namespace) -> None:
    with pydicom.dcmread(args.file) as dcm:
        dcm = drop_empty_tags(dcm)
        add_patient_age(dcm)
        drop_fields_by_length(dcm, args.max_length, inplace=True)

        if args.output == "txt":
            print(dcm)
        elif args.output == "json":
            print(json.dumps(dicom_to_json(dcm), indent=2))
