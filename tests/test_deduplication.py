from src.services.deduplication import deduplicate
from tests.conftest import make_job


def test_same_title_different_job_ids_both_remain() -> None:
    a = make_job(job_id="123", job_title="Software Engineer")
    b = make_job(job_id="456", job_title="Software Engineer")
    unique, duplicates, historical = deduplicate([a, b])
    assert len(unique) == 2
    assert duplicates == []
    assert historical == []


def test_same_id_is_duplicate() -> None:
    a = make_job(job_id="123", source="greenhouse")
    b = make_job(job_id="123", source="jobright")
    unique, duplicates, _ = deduplicate([a, b])
    assert len(unique) == 1
    assert len(duplicates) == 1


def test_missing_id_falls_back_to_fingerprint() -> None:
    a = make_job(job_id=None, job_title="Backend Engineer", location="Austin, TX")
    b = make_job(job_id=None, job_title="Backend Engineer", location="Austin, TX")
    unique, duplicates, _ = deduplicate([a, b])
    assert len(unique) == 1
    assert len(duplicates) == 1


def test_same_job_across_sources_is_one_row() -> None:
    from src.agents.discovery_agent import _dedupe_raw
    from src.models.job import RawJobPosting

    a = RawJobPosting(
        source="jobright",
        company_name="Acme Robotics",
        title="Software Engineer",
        job_id="1001",
        apply_url="https://boards.greenhouse.io/acmerobotics/jobs/1001",
    )
    b = RawJobPosting(
        source="greenhouse",
        company_name="Acme Robotics",
        title="Software Engineer",
        job_id="1001",
        apply_url="https://boards.greenhouse.io/acmerobotics/jobs/1001?gh_src=xyz",
    )
    unique = _dedupe_raw([a, b])
    assert len(unique) == 1


def test_historical_keys_are_skipped() -> None:
    job = make_job(job_id="123")
    unique, _, historical = deduplicate([job], known_keys={job.dedup_key})
    assert unique == []
    assert len(historical) == 1
    assert historical[0].is_new is False
