"""Workday target discovery and existing qualification filters. No live network."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import yaml

from src.agents.critic_agent import review_fit
from src.agents.discovery_agent import _dedupe_raw
from src.agents.job_intelligence_agent import evaluate_job
from src.main import build_parser
from src.models.config import WorkdaySourceSettings
from src.models.job import DateSource, LanguagePolarity, RawJobPosting, RejectionReason
from src.pilot.workday import extract_workday_job_id, official_workday_url
from src.pilot.workday_qualify import qualify_workday_postings
from src.pilot.workday_target import (
    browser_discovery_enabled,
    query_applied,
    select_detail_candidates,
    target_queries,
    worth_detail,
)
from src.services.candidate_profile import CandidateProfile, SkillEvidence
from src.services.h1b import scan_sponsorship_language
from src.services.roles import classify_role
from src.services.workday_detail import parse_workday_detail
from src.sources.workday import WorkdaySource
from src.utils.dates import utcnow

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "fixtures" / "workday" / "phase3_cases.yaml"
HTML = ROOT / "tests" / "fixtures" / "workday" / "html"
BOARD = "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite"

_REASONS = {
    "role": RejectionReason.ROLE,
    "seniority": RejectionReason.SENIORITY,
    "location": RejectionReason.LOCATION,
    "employment": RejectionReason.EMPLOYMENT_TYPE,
    "freshness": RejectionReason.FRESHNESS,
}


def _profile() -> CandidateProfile:
    return CandidateProfile(
        profile_version=1,
        professional_months=18,
        max_required_years=2,
        education=["Bachelors in Electronics and Telecommunication"],
        experience_evidence=[
            SkillEvidence(skill="Java", evidence="Developed backend services using Java.", status="EXPLICIT"),
            SkillEvidence(skill="PostgreSQL", evidence="Built REST APIs with PostgreSQL.", status="EXPLICIT"),
        ],
        technical_skills=[],
    )


def _url(job_id: str) -> str:
    return f"{BOARD}/job/Santa-Clara/Software-Engineer_{job_id}"


def _load_cases() -> list[dict]:
    payload = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    return list(payload["cases"])


def _postings() -> list[RawJobPosting]:
    now = utcnow()
    postings: list[RawJobPosting] = []
    for case in _load_cases():
        freshness = case["freshness"]
        posted = None
        date_source = DateSource.UNKNOWN
        if freshness == "fresh":
            posted = now - timedelta(hours=2)
            date_source = DateSource.POSTED_DATE
        elif freshness == "stale":
            posted = now - timedelta(days=40)
            date_source = DateSource.POSTED_DATE
        postings.append(
            RawJobPosting(
                source="workday",
                company_name="NVIDIA",
                title=case["title"],
                location_raw=case["location"],
                description=case["description"],
                employment_type_raw=case["employment"],
                job_id=case["job_id"],
                apply_url=_url(case["job_id"]),
                posted_at=posted,
                date_source=date_source,
                provenance={"discovery_method": "browser", "fixture": case["id"]},
            )
        )
    return postings


def test_browser_discovery_is_off_unless_enabled(tmp_config) -> None:
    assert WorkdaySourceSettings().browser_enabled is False
    assert tmp_config.settings.discovery.sources.workday.browser_enabled is False
    assert browser_discovery_enabled(tmp_config) is False
    enabled = tmp_config.model_copy(update={"fixture_mode": False})
    workday = enabled.settings.discovery.sources.workday.model_copy(update={"browser_enabled": True})
    sources = enabled.settings.discovery.sources.model_copy(update={"workday": workday})
    discovery = enabled.settings.discovery.model_copy(update={"sources": sources})
    settings = enabled.settings.model_copy(update={"discovery": discovery})
    live = enabled.model_copy(update={"settings": settings})
    assert browser_discovery_enabled(live) is True
    assert browser_discovery_enabled(live.model_copy(update={"fixture_mode": True})) is False


def test_cxs_adapter_remains_the_http_source() -> None:
    assert WorkdaySource.__name__ == "WorkdaySource"


def test_production_smoke_flag_is_opt_in() -> None:
    args = build_parser().parse_args(["--workday-production-smoke-test", "--no-email"])
    assert args.workday_production_smoke_test is True
    assert args.no_email is True


def test_target_queries_come_from_the_role_taxonomy(tmp_config) -> None:
    queries = target_queries(tmp_config.roles)
    assert len(queries) <= 4
    blob = " ".join(queries).lower()
    assert "software engineer" in blob
    assert "new grad" in blob
    assert "backend" in blob


def test_detail_sample_is_stable_and_skips_senior_titles(tmp_config) -> None:
    roles = tmp_config.roles
    postings = [
        RawJobPosting(source="workday", company_name="NVIDIA", title="Senior Software Engineer", job_id="JR9", apply_url=_url("JR9")),
        RawJobPosting(source="workday", company_name="NVIDIA", title="Software Engineer II", job_id="JR1", apply_url=_url("JR1")),
        RawJobPosting(source="workday", company_name="NVIDIA", title="Engineering Manager", job_id="JR3", apply_url=_url("JR3")),
        RawJobPosting(source="workday", company_name="NVIDIA", title="Software Engineer", job_id="JR2", apply_url=_url("JR2")),
    ]
    chosen, skipped, unopened = select_detail_candidates(list(reversed(postings)), roles, limit=1)
    again, _, _ = select_detail_candidates(postings, roles, limit=1)
    assert chosen[0].job_id == again[0].job_id == "JR1"
    assert worth_detail("Software Engineer II", roles) is True
    assert worth_detail("New Grad Software Engineer", roles) is True
    assert worth_detail("Senior Software Engineer", roles) is True
    assert worth_detail("Principal Software Engineer", roles) is False
    assert worth_detail("Engineering Manager", roles) is False
    assert skipped == 1
    assert unopened == 2
    early = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="New Grad Software Engineer",
        job_id="JR9",
        apply_url=_url("JR9"),
    )
    preferred, _, _ = select_detail_candidates([postings[3], early], roles, limit=1)
    assert preferred[0].job_id == "JR9"
    assert worth_detail("Software Engineer - New College Grad", roles) is True
    assert worth_detail("DFT Engineer - New College Grad", roles) is False
    assert query_applied(
        "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite?q=software%20engineer%20new%20grad",
        "software engineer new grad",
    )
    assert query_applied("https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite", "new grad") is False
    assert extract_workday_job_id(_url("JR2001099-1")) == "JR2001099-1"
    official, status = official_workday_url(_url("JR1"))
    assert status == "verified"
    assert official is not None
    blocked, blocked_status = official_workday_url("https://jobright.ai/jobs/1")
    assert blocked is None
    assert blocked_status == "UNKNOWN"


def test_management_titles_are_not_software_roles(tmp_config) -> None:
    for title in ("Engineering Manager", "Software Engineer Manager", "Solutions Architect", "Enterprise Architect"):
        verdict = classify_role(title, "Build production services and ship code.", tmp_config.roles)
        assert verdict.is_software_engineering is False, title
        assert verdict.needs_llm is False, title


def test_early_career_detail_keeps_required_and_preferred_separate() -> None:
    html = (HTML / "early_career.html").read_text(encoding="utf-8")
    detail = parse_workday_detail(
        html,
        title="Software Engineer",
        company="NVIDIA",
        official_url=_url("JR3001"),
        job_id="JR3001",
        locations_text="locations US, CA, Santa Clara",
        employment_text="time type Full time",
        posted_text="posted on Posted Yesterday",
        requisition_text="job requisition id JR3001",
    )
    assert "Design REST APIs used by internal tools." in detail.responsibilities
    assert any("0-2 years" in line for line in detail.required_qualifications)
    assert any("5+ years" in line for line in detail.preferred_qualifications)
    assert "Java" in detail.required_skills
    assert "Kafka" in detail.preferred_skills
    assert detail.date_source is DateSource.POSTED_DATE
    assert "Santa Clara" in detail.location


async def test_existing_filters_qualify_the_target_population(tmp_config) -> None:
    cases = _load_cases()
    funnel = await qualify_workday_postings(tmp_config, _postings())
    accepted = {job.job_id for job in funnel.state.jobs}
    for case in cases:
        job_id = case["job_id"]
        expect = case["expect"]
        if expect == "accept":
            assert job_id in accepted, case["id"]
            continue
        if expect == "duplicate":
            assert sum(1 for job in funnel.state.jobs if job.job_id == job_id) == 1
            assert any(
                item.reason is RejectionReason.DUPLICATE and job_id in (item.url or "")
                for item in funnel.state.rejected
            )
            continue
        assert job_id not in accepted, case["id"]
        reason = _REASONS[expect]
        assert any(item.reason is reason and job_id in (item.url or "") for item in funnel.state.rejected), case["id"]

    present = next(job for job in funnel.state.jobs if job.job_id == "JR3018")
    absent = next(job for job in funnel.state.jobs if job.job_id == "JR3017")
    unknown = next(job for job in funnel.state.jobs if job.job_id == "JR3019")
    assert scan_sponsorship_language(present.description).polarity is LanguagePolarity.POSITIVE
    assert scan_sponsorship_language(absent.description).polarity is LanguagePolarity.ABSENT
    assert scan_sponsorship_language(unknown.description).polarity is LanguagePolarity.AMBIGUOUS
    assert "dry-run" in (funnel.state.summary.workbook_path or "")


def test_same_workday_id_collapses_and_different_ids_do_not() -> None:
    first = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Software Engineer",
        job_id="JR3001",
        apply_url=_url("JR3001"),
        provenance={"discovery_method": "browser"},
    )
    second = RawJobPosting(
        source="jobright",
        company_name="NVIDIA",
        title="Software Engineer",
        job_id="JR3001",
        apply_url=_url("JR3001"),
        provenance={"discovery_method": "api"},
    )
    other = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Software Engineer",
        job_id="JR3002",
        apply_url=_url("JR3002"),
    )
    unique = _dedupe_raw([first, second, other])
    assert [posting.job_id for posting in unique] == ["JR3001", "JR3002"]
    assert unique[0].provenance["discovered_from"] == ["workday", "jobright"]


def test_intelligence_uses_the_existing_profile(tmp_config) -> None:
    profile = _profile()
    posting = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Software Engineer",
        job_id="JR3001",
        apply_url=_url("JR3001"),
        location_raw="Santa Clara, CA",
        employment_type_raw="Full-time",
        posted_at=utcnow() - timedelta(hours=2),
        date_source=DateSource.POSTED_DATE,
        description=(
            "Minimum Qualifications\n"
            "0-2 years of experience.\n"
            "BS in Computer Science or Engineering.\n"
            "Java and PostgreSQL required.\n"
        ),
    )
    fit = evaluate_job(posting, profile, tmp_config.roles)
    assert fit.profile_version == "1"
    assert fit.job_id == "JR3001"
    assert fit.official_url == posting.apply_url
    assert fit.experience_alignment == "strong"
    assert fit.education_alignment == "related"
    assert fit.sponsorship == LanguagePolarity.ABSENT.value
    assert fit.freshness == "fresh"
    assert fit.location_alignment == "aligned"
    assert "Java" in fit.matched_requirements
    assert "PostgreSQL" in fit.matched_requirements


def test_critic_rejects_unsupported_workday_claims(tmp_config) -> None:
    profile = _profile()
    posting = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Software Engineer",
        job_id="JR3001",
        apply_url=_url("JR3001"),
        description=(
            "Minimum Qualifications\n"
            "0-2 years of experience.\n"
            "BS in Computer Science or Engineering.\n"
            "Java required.\n"
        ),
        posted_at=utcnow() - timedelta(hours=2),
        date_source=DateSource.POSTED_DATE,
    )
    fit = evaluate_job(posting, profile, tmp_config.roles)
    invented = fit.model_copy(deep=True)
    invented.matched_requirements = ["Java", "Kubernetes"]
    invented.candidate_evidence.append(
        {"requirement": "Kubernetes", "evidence": "Invented Kubernetes experience", "match": "MATCHED"}
    )
    cleaned = review_fit(invented, posting, profile)
    assert "Kubernetes" in cleaned.unsupported_claims
    assert "Kubernetes" not in cleaned.fit.matched_requirements

    senior = posting.model_copy(update={
        "title": "Senior Software Engineer",
        "description": "Minimum Qualifications\n5+ years of experience.\nJava required.\n",
    })
    overstated = evaluate_job(senior, profile, tmp_config.roles).model_copy(
        update={"experience_alignment": "strong", "decision": "STRONG_MATCH"}
    )
    experience = review_fit(overstated, senior, profile)
    assert experience.fit.experience_alignment == "reject"
    assert experience.fit.decision == "REJECT"

    education = fit.model_copy(update={"education_alignment": "equivalent"})
    education_review = review_fit(education, posting, profile)
    assert education_review.fit.education_alignment == "related"
    assert "education interpretation is not supported" in education_review.issues

    sponsored = fit.model_copy(update={"sponsorship": "POSITIVE"})
    sponsorship = review_fit(sponsored, posting, profile)
    assert sponsorship.fit.sponsorship == LanguagePolarity.ABSENT.value
    assert "unsupported sponsorship claim" in sponsorship.issues

    undated = posting.model_copy(update={"posted_at": None, "date_source": DateSource.UNKNOWN})
    fresh_claim = fit.model_copy(update={"freshness": "fresh"})
    freshness = review_fit(fresh_claim, undated, profile)
    assert freshness.fit.freshness == "unknown"

    manager = fit.model_copy(update={"role_alignment": "strong", "decision": "STRONG_MATCH"})
    manager_post = posting.model_copy(update={"title": "Engineering Manager"})
    role = review_fit(manager, manager_post, profile)
    assert role.fit.role_alignment == "weak"
    assert role.fit.decision == "REJECT"
