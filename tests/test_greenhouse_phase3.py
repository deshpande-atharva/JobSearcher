"""Phase 3 matching, freshness, and Greenhouse metadata tests. No live network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.models.job import DateSource, RawJobPosting
from src.pilot.greenhouse import enrich_browser_jobs
from src.services.candidate_profile import CandidateProfile, SkillEvidence
from src.services.freshness import is_fresh
from src.services.job_requirements import extract_job_requirements
from tests.conftest import make_job


def _profile() -> CandidateProfile:
    return CandidateProfile(
        profile_version=1,
        professional_months=18,
        max_required_years=2,
        education=["M.S. in Software Engineering", "Bachelors in Computer Engineering"],
        experience_evidence=[
            SkillEvidence(skill="Java", evidence="Developed backend services using Java.", status="EXPLICIT"),
            SkillEvidence(skill="Spring Boot", evidence="Built services with Spring Boot.", status="EXPLICIT"),
            SkillEvidence(
                skill="Java",
                evidence="Built services with Spring Boot.",
                status="INFERRED",
            ),
            SkillEvidence(skill="REST APIs", evidence="Built REST APIs with PostgreSQL.", status="EXPLICIT"),
            SkillEvidence(skill="PostgreSQL", evidence="Built REST APIs with PostgreSQL.", status="EXPLICIT"),
            SkillEvidence(skill="AWS", evidence="Provisioned AWS infrastructure using Terraform.", status="EXPLICIT"),
            SkillEvidence(skill="React", evidence="Built the React management experience.", status="EXPLICIT"),
        ],
    )


def _post(title: str, description: str, **kwargs: object) -> RawJobPosting:
    return RawJobPosting(
        source="greenhouse",
        company_name="Acme",
        title=title,
        description=description,
        job_id="8184174",
        apply_url="https://careers.example.com/positions/8184174/",
        location_raw="Seattle, WA",
        **kwargs,
    )


def test_required_and_preferred_skills_stay_separate() -> None:
    parsed = extract_job_requirements("Experience with Java required; Kafka is a plus.")
    assert parsed.required_skills == ["Java"]
    assert parsed.preferred_skills == ["Kafka"]
    assert "Kafka" not in parsed.required_skills


def test_strong_partial_and_missing_skills(tmp_config) -> None:
    profile = _profile()
    strong = evaluate_job(
        _post("Backend Engineer", "Java required. Spring Boot required. REST APIs required. PostgreSQL preferred."),
        profile,
        tmp_config.roles,
    )
    assert strong.decision == "STRONG_MATCH"
    assert "Java" in strong.matched_requirements
    assert "Kafka" not in strong.matched_requirements
    assert strong.profile_version == "1"
    assert strong.job_id == "8184174"
    assert strong.official_url.endswith("/8184174/")
    java = next(item for item in strong.candidate_evidence if item["requirement"] == "Java")
    assert java["match"] == "MATCHED"
    assert java["evidence_type"] in {"EXPLICIT", "INFERRED"}
    assert "Java" in java["resume_evidence"] or "Spring Boot" in java["resume_evidence"]

    partial = evaluate_job(
        _post("Backend Engineer", "Java required. Kubernetes required."),
        profile,
        tmp_config.roles,
    )
    assert "Java" in partial.matched_requirements
    assert "Kubernetes" in partial.missing_requirements
    assert "Kubernetes" not in partial.matched_requirements
    assert partial.technical_alignment == "partial"

    preferred = evaluate_job(
        _post("Backend Engineer", "Java required. Kafka is a plus."),
        profile,
        tmp_config.roles,
    )
    assert "Kafka" in preferred.preferred_requirements
    assert "Kafka" not in preferred.missing_requirements
    assert "Kafka" not in preferred.matched_requirements


def test_experience_uses_professional_months(tmp_config) -> None:
    profile = _profile()
    early = evaluate_job(
        _post("Software Engineer", "0-2 years of experience. Java required."),
        profile,
        tmp_config.roles,
    )
    one_year = evaluate_job(
        _post("Software Engineer", "1+ years of experience required. Java required."),
        profile,
        tmp_config.roles,
    )
    three = evaluate_job(
        _post("Software Engineer", "3+ years required. Java required."),
        profile,
        tmp_config.roles,
    )
    preferred = evaluate_job(
        _post("Software Engineer", "Java required. 5+ years preferred."),
        profile,
        tmp_config.roles,
    )
    assert early.experience_alignment == "strong"
    assert early.decision != "REJECT"
    assert one_year.experience_alignment == "strong"
    assert three.decision == "REJECT"
    assert three.experience_alignment == "reject"
    assert preferred.decision != "REJECT"
    assert preferred.experience_alignment != "reject"


def test_ambiguous_and_nonswe_titles_are_not_title_matches(tmp_config) -> None:
    profile = _profile()
    ambiguous = evaluate_job(
        _post("Business Systems Engineer", "Partner with stakeholders on business process requirements."),
        profile,
        tmp_config.roles,
    )
    nonswe = evaluate_job(
        _post("Account Manager", "Manage client accounts and sales targets."),
        profile,
        tmp_config.roles,
    )
    assert ambiguous.decision in {"UNKNOWN", "POSSIBLE_MATCH", "REJECT"}
    assert ambiguous.decision != "STRONG_MATCH"
    assert nonswe.decision == "REJECT"
    assert nonswe.role_alignment == "weak"


def test_related_degree_is_not_called_computer_science(tmp_config) -> None:
    profile = _profile().model_copy(
        update={"education": ["Bachelors in Electronics and Telecommunication Engineering"]}
    )
    fit = evaluate_job(
        _post(
            "Software Engineer",
            "Bachelor's degree in Computer Science or related field required. Java required.",
        ),
        profile,
        tmp_config.roles,
    )
    assert fit.education_alignment == "related"
    assert fit.education_alignment != "aligned"


def test_unknown_sponsorship_and_freshness(tmp_config) -> None:
    profile = _profile()
    fit = evaluate_job(
        _post("Software Engineer", "Java required.", date_source=DateSource.UNKNOWN),
        profile,
        tmp_config.roles,
    )
    assert fit.freshness == "unknown"
    assert fit.sponsorship == "ABSENT"
    unknown_clock = datetime(2026, 9, 25, 12, tzinfo=UTC)
    discovered_today = make_job(posted_at=None, updated_at=None)
    discovered_today.date_source = DateSource.UNKNOWN
    assert is_fresh(discovered_today, 24, now=unknown_clock)[0] is False


def test_freshness_cases_use_injected_clock() -> None:
    now = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    assert is_fresh(make_job(posted_at=now - timedelta(hours=2)), 24, now=now)[0] is True
    assert is_fresh(make_job(posted_at=now - timedelta(hours=23)), 24, now=now)[0] is True
    assert is_fresh(make_job(posted_at=now - timedelta(hours=24, seconds=1)), 24, now=now)[0] is False
    assert is_fresh(make_job(posted_at=now - timedelta(days=3)), 24, now=now)[0] is False
    updated = make_job(posted_at=None, updated_at=now - timedelta(hours=2), date_source=DateSource.UPDATED_DATE)
    assert is_fresh(updated, 24, now=now)[0] is True
    unknown = make_job(posted_at=None, updated_at=None)
    unknown.date_source = DateSource.UNKNOWN
    assert is_fresh(unknown, 24, now=now)[0] is False


def test_enrichment_preserves_browser_discovery_and_ids() -> None:
    browser_job = RawJobPosting(
        source="greenhouse",
        company_name="GitLab",
        title="Backend Engineer",
        job_id="42",
        apply_url="https://job-boards.greenhouse.io/gitlab/jobs/42",
        date_source=DateSource.UNKNOWN,
        provenance={"discovery_method": "browser", "discovered_from": ["greenhouse"]},
    )
    api_only = RawJobPosting(source="greenhouse", company_name="GitLab", title="Not seen", job_id="99")
    structured = RawJobPosting(
        source="greenhouse",
        company_name="GitLab",
        title="Backend Engineer",
        job_id="42",
        description="Java required.",
        posted_at=datetime(2026, 9, 25, 10, 0, tzinfo=UTC),
        date_source=DateSource.POSTED_DATE,
        provenance={"discovery_method": "structured"},
    )
    enriched = enrich_browser_jobs([browser_job], [structured, api_only])
    assert len(enriched) == 1
    assert enriched[0].job_id == "42"
    assert enriched[0].provenance["discovery_method"] == "browser"
    assert enriched[0].date_source is DateSource.POSTED_DATE
    assert enriched[0].description == "Java required."
    assert enriched[0].provenance["metadata_source"] == "greenhouse_structured"


def test_critic_removes_unsupported_skill_and_experience(tmp_config) -> None:
    profile = _profile()
    posting = _post("Software Engineer", "Java required. 3+ years required.")
    fit = evaluate_job(posting, profile, tmp_config.roles)
    invented = fit.model_copy(deep=True)
    invented.decision = "STRONG_MATCH"
    invented.experience_alignment = "strong"
    invented.matched_requirements = ["Java", "Kafka"]
    invented.candidate_evidence.append(
        {"requirement": "Kafka", "evidence": "Invented Kafka experience", "match": "MATCHED"}
    )
    cleaned = review_fit(invented, posting, profile)
    assert "Kafka" in cleaned.unsupported_claims
    assert "Kafka" not in cleaned.fit.matched_requirements
    assert cleaned.fit.experience_alignment == "reject"
    assert cleaned.fit.profile_version == "1"
    again = evaluate_job(posting, profile, tmp_config.roles)
    assert again.profile_version == fit.profile_version
    assert again.job_id == fit.job_id


def test_escaped_greenhouse_html_splits_required_and_preferred() -> None:
    from src.utils.normalization import html_to_text

    raw = (
        "&lt;h2&gt;What you'll bring&lt;/h2&gt;&lt;ul&gt;"
        "&lt;li&gt;Experience with Java required&lt;/li&gt;&lt;/ul&gt;"
        "&lt;p&gt;&lt;strong&gt;Preferred requirements&lt;/strong&gt;&lt;/p&gt;&lt;ul&gt;"
        "&lt;li&gt;Kafka is a plus&lt;/li&gt;&lt;/ul&gt;"
    )
    parsed = extract_job_requirements(html_to_text(raw))
    assert parsed.required_skills == ["Java"]
    assert parsed.preferred_skills == ["Kafka"]


def test_mentoring_juniors_and_internal_are_not_entry_level(tmp_config) -> None:
    profile = _profile()
    posting = _post(
        "Senior AI Engineer",
        "What you'll bring\n"
        "Hands-on technical leader delivering internal AI solutions.\n"
        "Preferred requirements\n"
        "Experience mentoring junior engineers.\n"
        "Java required.",
    )
    fit = evaluate_job(posting, profile, tmp_config.roles)
    assert fit.experience_alignment != "strong"
    assert fit.decision != "REJECT" or fit.hard_filter != "experience"
