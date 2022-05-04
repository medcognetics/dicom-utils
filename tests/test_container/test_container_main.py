#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from dicom_utils import DicomFactory
from dicom_utils.container.__main__ import run
from dicom_utils.container.record import DicomFileRecord


class TestSymlinkPipeline:
    @pytest.mark.parametrize("implants", [False, True])
    @pytest.mark.parametrize("spot", [False, True])
    def test_case_dir_structure(self, tmp_path, implants, spot):
        types = ("ffdm", "tomo", "synth", "ultrasound")
        factory = DicomFactory.complete_mammography_case_factory(
            types=types,
            spot_compression=spot,
            implants=implants,
        )
        paths = []
        for i in range(num_cases := 3):
            case_dir = Path(tmp_path, f"Original-{i}")
            case_dir.mkdir(parents=True)
            dicoms = factory(StudyInstanceUID=f"study-{i}")
            outputs = DicomFactory.save_dicoms(case_dir, dicoms)
            paths.append(outputs)
        dest = Path(tmp_path, "symlinks")
        dest.mkdir()
        result = run(tmp_path, dest, threads=True, use_bar=False)
        for output, paths in result.items():
            recs = {k: [p.relative_to(dest) for p in v] for k, v in paths.items()}
            assert len(recs) == num_cases
            assert all(v for v in recs.values())

    def test_longitudinal_case_dir_structure(self, tmp_path):
        types = ("ffdm", "tomo", "synth", "ultrasound")
        factory = DicomFactory.complete_mammography_case_factory(
            types=types,
        )
        paths = []
        base_year = 2010
        num_years = 3
        for i in range(num_cases := 3):
            case_dir = Path(tmp_path, f"Original-{i}")
            for j in range(num_years):
                year_dir = Path(case_dir, str(base_year + j))
                year_dir.mkdir(parents=True)
                study_uid = f"study-{i*num_years + j}"
                patient_id = f"patient-{i}"
                dicoms = factory(
                    StudyInstanceUID=study_uid,
                    PatientID=patient_id,
                )
                outputs = DicomFactory.save_dicoms(year_dir, dicoms)
                paths.append(outputs)
        dest = Path(tmp_path, "symlinks")
        dest.mkdir()
        helpers = ["study-date-from-path"]

        result = run(tmp_path, dest, helpers=helpers, threads=True, use_bar=False)
        for output, paths in result.items():
            recs = {k: [p.relative_to(dest) for p in v] for k, v in paths.items()}
            assert len(recs) == num_cases * num_years
            assert all(v for v in recs.values())
            assert all(
                not isinstance(r, DicomFileRecord) or (base_year <= r.year < base_year + num_years)
                for r in recs.values()
            )
