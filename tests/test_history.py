from pathlib import Path

from src.models.job import AppliedFlag, ApplicationStatus
from src.services.history import load_history
from src.services.xlsx import apply_tracking, write_workbooks
from tests.conftest import make_job


def test_history_reads_current_and_archive(tmp_config, tmp_path: Path) -> None:
    job = make_job(job_id="hist-1")
    job.applied = AppliedFlag.APPLIED
    job.status = ApplicationStatus.OA
    current = tmp_path / "data" / "current" / "jobs.xlsx"
    archive = tmp_path / "data" / "archive"
    write_workbooks([job], current_path=current, archive_dir=archive, write_archive=True)

    index = load_history(tmp_config)
    assert job.dedup_key in index.known_keys
    assert index.tracking[job.dedup_key]["applied"] == AppliedFlag.APPLIED.value
    assert index.tracking[job.dedup_key]["status"] == ApplicationStatus.OA.value


def test_apply_tracking_does_not_invent_values() -> None:
    job = make_job()
    apply_tracking(job, {})
    assert job.applied is AppliedFlag.NOT_APPLIED
    assert job.status is ApplicationStatus.NOT_STARTED
