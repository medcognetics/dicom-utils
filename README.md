# DICOM Utils

Collection of helpful scripts and Python methods for working with DICOMs.

## Setup

This repo can be installed with `pip`. To install to a virtual environment:
1. Run `make init` to create a virtual environment with dicom-utils installed.
2. Call utilities with `venv/bin/python -m dicom_utils`

## Usage

The following scripts are provided:
  * `cat` - Print DICOM metadata output as text or JSON
  * `find` - Find valid DICOM files, with options to filter by image type
  * `dicom2img` - Convert DICOM to static image or GIF
