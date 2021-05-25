# DICOM Utils

Collection of helpful scripts and Python methods for working with DICOMs.

## Setup

This repo can be installed with `pip`. To install to a virtual environment:
1. Run `make init` to create a virtual environment with dicom-utils installed.
2. Call utilities with `venv/bin/python -m dicom_utils`

Alternatively, install the repo without a virtual environment and run the 
entrypoints provided by setup.py
1. `pip install .` or `pip install -e .`
2. Run utilities anywhere as `dicomcat`, `dicomfind`, etc.

## Usage

The following scripts are provided:
  * `dicomcat` - Print DICOM metadata output as text or JSON
  * `dicomfind` - Find valid DICOM files, with options to filter by image type
  * `dicom2img` - Convert DICOM to static image or GIF
  * `dicom_types` - Print unique values of the "Image Type" field
  * `dicom_overlap` - Find StudyInstanceUID values shared by files in two directories
