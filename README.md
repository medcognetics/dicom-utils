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
  * `dicomphi` - Find and overwrite PHI across DICOM files
  * `dicom2img` - Convert DICOM to static image or GIF
  * `dicom_types` - Print unique values of the "Image Type" field
  * `dicom_overlap` - Find StudyInstanceUID values shared by files in two directories

## PHI Anonymization Rules
If anonymization is enabled when running `dicomphi`, fields defined in
[this script](https://github.com/medcognetics/dicom-anonymizer/blob/master/dicomanonymizer/dicomfields.py)
are anonymized
with the exception of fields which are affected by additional rules located 
[here](https://github.com/medcognetics/dicom-utils/blob/master/dicom_utils/anonymize.py).

## pynvjpeg

To install [pynvjpeg](https://github.com/medcognetics/pynvjpeg2k) for accelerated JPEG2000 decoding,
install the `j2k` extra.

```bash
$ pip install -e ".[j2k,dev]"
```

The following steps may be required:

1. `apt install cmake`

2. Add CUDA tools to path

```bash
CUDA_FOLDER="cuda-12.0"
export PATH="/usr/local/$CUDA_FOLDER/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/$CUDA_FOLDER/lib64:$LD_LIBRARY_PATH"
```

3. `apt -y install python3-pybind11`
