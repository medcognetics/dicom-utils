#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from dicom_utils.cli.organize import organize
from dicom_utils.dicom_factory import CompleteMammographyStudyFactory, DicomFactory


class TestSymlinkPipeline:
    @pytest.mark.parametrize("implants", [False, True])
    @pytest.mark.parametrize("spot", [False, True])
    def test_case_dir_structure(self, tmp_path, implants, spot):
        types = ("ffdm", "tomo", "synth", "ultrasound")
        factory = CompleteMammographyStudyFactory(
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
        result = organize(tmp_path, dest, threads=False, use_bar=False, jobs=0)
        for output, results in result.items():
            recs = {k: [r.path.relative_to(dest) for r in v] for k, v in results.items()}
            assert len(recs) == num_cases
            assert all(v for v in recs.values())
