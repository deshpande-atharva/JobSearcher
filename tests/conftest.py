"""Shared fixtures. Tests never touch live websites."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.models.config import AppConfig, load_config
from src.models.job import (
    DateSource,
    EmploymentType,
    Job,
    RemoteType,
    VisaSponsorshipStatus,
)
from src.utils.dates import parse_datetime

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def tmp_config(tmp_path: Path) -> AppConfig:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy(ROOT / "config" / "settings.yaml", config_dir / "settings.yaml")
    shutil.copy(ROOT / "config" / "roles.yaml", config_dir / "roles.yaml")
    (config_dir / "companies.yaml").write_text(
        """
defaults: {}
companies:
  - name: Acme Robotics
    aliases: ["Acme Robotics Inc"]
    ats_type: greenhouse
    ats_identifier: acmerobotics
    careers_url: https://boards.greenhouse.io/acmerobotics
  - name: Northwind Labs
    ats_type: lever
    ats_identifier: northwindlabs
    careers_url: https://jobs.lever.co/northwindlabs
""".strip()
        + "\n",
        encoding="utf-8",
    )
    fixtures = ROOT / "tests" / "fixtures"
    return load_config(
        config_dir,
        env={},
        dry_run=True,
        fixture_mode=True,
        send_email=False,
        overrides={"fixtures": {"directory": str(fixtures)}, "output": {"current_workbook": str(tmp_path / "data" / "current" / "jobs.xlsx"), "archive_dir": str(tmp_path / "data" / "archive")}},
    )


def make_job(**overrides: object) -> Job:
    posted = overrides.pop("posted_at", parse_datetime("2 hours ago"))
    data = dict(
        company="Acme Robotics",
        job_title="Software Engineer",
        normalized_role="Software Engineer",
        role_family="SOFTWARE_ENGINEER",
        location="Seattle, WA",
        location_city="Seattle",
        location_state="WA",
        remote_type=RemoteType.ONSITE,
        employment_type=EmploymentType.FULL_TIME,
        posted_at=posted,
        date_source=DateSource.POSTED_DATE,
        job_id="1001",
        source="greenhouse",
        direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/1001",
        description="New graduates welcome. 0-2 years. Write production code.",
        visa_sponsorship_status=VisaSponsorshipStatus.UNKNOWN,
    )
    data.update(overrides)
    return Job.model_validate(data)


@pytest.fixture
def sample_job() -> Job:
    return make_job()
