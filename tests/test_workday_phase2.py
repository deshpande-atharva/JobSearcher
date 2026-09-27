"""Workday detail extraction and existing intelligence integration. No live network."""

from __future__ import annotations

from pathlib import Path

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.models.job import DateSource, RawJobPosting
from src.pilot.workday_intelligence import select_sample
from src.services.candidate_profile import CandidateProfile, SkillEvidence
from src.services.workday_detail import parse_workday_detail

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "tests" / "fixtures" / "workday" / "html"
URL = "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Santa-Clara/Backend-Engineer_JR1001"


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
    )


def _detail(name: str, **kwargs):
    html = (HTML / name).read_text(encoding="utf-8")
    return parse_workday_detail(
        html,
        title="Backend Engineer",
        company="NVIDIA",
        official_url=URL,
        job_id="JR1001",
        locations_text=kwargs.get("locations", "locations US, CA, Santa Clara"),
        employment_text=kwargs.get("employment", "time type Full time"),
        posted_text=kwargs.get("posted", ""),
        requisition_text="job requisition id JR1001",
    )


def test_detail_html_splits_required_and_preferred() -> None:
    detail = _detail("job_detail.html")
    assert "Design REST APIs used by internal tools." in detail.responsibilities
    assert any("Java" in line and "PostgreSQL" in line for line in detail.required_qualifications)
    assert any("Kafka" in line for line in detail.preferred_qualifications)
    assert "Java" in detail.required_skills
    assert "PostgreSQL" in detail.required_skills
    assert "Kafka" in detail.preferred_skills
    assert "Kafka" not in detail.required_skills
    assert detail.experience_requirements
    assert "Computer Science" in detail.education_requirements
    assert detail.employment_type.lower().startswith("full")
    assert "Santa Clara" in detail.location
    assert detail.sponsorship_language == "ABSENT"
    assert detail.clearance_requirements == "UNKNOWN"
    assert detail.date_source is DateSource.UNKNOWN


def test_escaped_html_is_decoded_before_headings() -> None:
    detail = _detail("job_detail_escaped.html")
    assert detail.required_qualifications
    assert "Java" in detail.required_skills
    assert "Kafka" in detail.preferred_skills
    assert "Sign In" not in detail.description


def test_multi_location_chrome_stays_unknown() -> None:
    detail = _detail("job_detail.html", locations="locations 5 Locations")
    assert detail.location == "UNKNOWN"
    assert detail.work_arrangement == "UNKNOWN"


def test_posted_yesterday_is_a_posted_date_not_discovery_time() -> None:
    detail = _detail("job_detail.html", posted="posted on Posted Yesterday")
    assert detail.date_source is DateSource.POSTED_DATE
    assert detail.posted_at_raw


def test_sample_is_stable_and_mixed() -> None:
    jobs = [
        RawJobPosting(source="workday", company_name="NVIDIA", title="Backend Engineer", job_id="JR2", apply_url=URL),
        RawJobPosting(source="workday", company_name="NVIDIA", title="Senior Software Engineer", job_id="JR3", apply_url=URL),
        RawJobPosting(source="workday", company_name="NVIDIA", title="Recruiting Manager", job_id="JR1", apply_url=URL),
    ]
    chosen = select_sample(jobs)
    ids = [item.job_id for item in chosen]
    assert ids[0] == "JR2"
    assert "JR3" in ids
    assert "JR1" in ids
    assert select_sample(jobs)[0].job_id == select_sample(list(reversed(jobs)))[0].job_id


def test_existing_intelligence_and_critic_use_extracted_posting(tmp_config) -> None:
    detail = _detail("job_detail.html")
    posting = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Backend Engineer",
        description=detail.description,
        location_raw=detail.location,
        employment_type_raw=detail.employment_type,
        job_id="JR1001",
        apply_url=URL,
        date_source=DateSource.UNKNOWN,
    )
    profile = _profile()
    fit = evaluate_job(posting, profile, tmp_config.roles)
    assert fit.profile_version == "1"
    assert fit.job_source == "workday"
    assert fit.job_id == "JR1001"
    assert "Java" in fit.matched_requirements
    assert "Kafka" not in fit.missing_requirements
    assert fit.experience_alignment == "reject"
    assert fit.decision == "REJECT"
    assert fit.education_alignment == "related"
    assert fit.sponsorship == "ABSENT"
    assert fit.freshness == "unknown"
    invented = fit.model_copy(deep=True)
    invented.decision = "STRONG_MATCH"
    invented.experience_alignment = "strong"
    invented.matched_requirements = ["Java", "Kubernetes"]
    invented.candidate_evidence.append(
        {"requirement": "Kubernetes", "evidence": "Invented cluster work", "match": "MATCHED"}
    )
    cleaned = review_fit(invented, posting, profile)
    assert "Kubernetes" in cleaned.unsupported_claims
    assert "Kubernetes" not in cleaned.fit.matched_requirements
    assert cleaned.fit.experience_alignment == "reject"
    assert cleaned.fit.decision == "REJECT"
