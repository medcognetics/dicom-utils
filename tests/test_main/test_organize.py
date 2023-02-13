#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import pytest
import tqdm_multiprocessing

from dicom_utils.cli.organize import organize
from dicom_utils.dicom_factory import CompleteMammographyStudyFactory, DicomFactory


class TestSymlinkPipeline:
    @pytest.mark.parametrize("implants", [False, True])
    @pytest.mark.parametrize("spot", [False, True])
    def test_case_dir_structure(self, tmp_path, implants, spot):
        types = ("ffdm", "tomo", "synth", "ultrasound")
        paths = []
        seen_sop_uids = []
        for i in range(num_cases := 3):
            factory = CompleteMammographyStudyFactory(
                types=types,
                spot_compression=spot,
                implants=implants,
                seed=i,
            )
            case_dir = Path(tmp_path, f"Original-{i}")
            case_dir.mkdir(parents=True)
            dicoms = factory(StudyInstanceUID=f"study-{i}")
            outputs = DicomFactory.save_dicoms(case_dir, dicoms)
            paths.append(outputs)
            seen_sop_uids = seen_sop_uids + [d.SOPInstanceUID for d in dicoms]

        # non-deterministic test failures will happen if the UIDs aren't unique
        assert len(seen_sop_uids) == len(set(seen_sop_uids))

        dest = Path(tmp_path, "symlinks")
        dest.mkdir()
        result = organize(tmp_path, dest, threads=False, use_bar=False, jobs=4)
        for _, results in result.items():
            recs = {k: [r.path.relative_to(dest) for r in v] for k, v in results.items()}
            assert len(recs) == num_cases
            assert all(v for v in recs.values())

    @pytest.mark.parametrize("threads", [True, False])
    @pytest.mark.parametrize("jobs", [4, 8])
    def test_forward_jobs_and_threads(self, tmp_path, mocker, threads, jobs):
        spy = mocker.spy(tqdm_multiprocessing.ConcurrentMapper, "__init__")

        paths = []
        for i in range(num_cases := 3):
            factory = CompleteMammographyStudyFactory(
                seed=i,
            )
            case_dir = Path(tmp_path, f"Original-{i}")
            case_dir.mkdir(parents=True)
            dicoms = factory(StudyInstanceUID=f"study-{i}")
            outputs = DicomFactory.save_dicoms(case_dir, dicoms)
            paths.append(outputs)

        dest = Path(tmp_path, "symlinks")
        dest.mkdir()
        # set use_bar = True to help debugging
        timeout = 2
        organize(tmp_path, dest, use_bar=False, jobs=jobs, threads=threads, timeout=timeout)
        assert spy.call_count == 3
        for i, call in enumerate(spy.call_args_list):
            if i not in (1, 2):
                assert call.args[0].threads == threads, f"call {i} failed"
            else:
                # Grouper has thread override because of deadlocks
                assert call.args[0].threads, f"call {i} failed"
            assert call.args[0].jobs == jobs, f"call {i} failed"
            assert call.args[0].timeout == timeout, f"call {i} failed"
