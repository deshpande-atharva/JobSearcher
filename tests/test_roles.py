from src.models.config import load_config
from src.services.roles import classify_role
from src.services.seniority import classify_seniority
from src.utils.normalization import detect_employment_type, extract_experience_requirement, normalize_location
from tests.conftest import ROOT


def _roles():
    return load_config(ROOT / "config", env={}, fixture_mode=True, send_email=False).roles


def test_equivalent_software_titles() -> None:
    roles = _roles()
    for title in (
        "Software Engineer",
        "Software Development Engineer",
        "SDE",
        "Backend Engineer",
        "Full-Stack Software Engineer",
        "Site Reliability Engineer",
        "SRE",
        "Platform Engineer",
        "Infrastructure Engineer",
        "Cloud Engineer",
        "Systems Engineer",
        "Product Engineer",
        "DevOps Engineer",
    ):
        verdict = classify_role(title, "Write production code and ship services.", roles)
        assert verdict.is_software_engineering, title


def test_excluded_non_software_title() -> None:
    roles = _roles()
    verdict = classify_role("Data Analyst", "Build dashboards and reports.", roles)
    assert verdict.is_software_engineering is False
    assert verdict.needs_llm is False


def test_data_engineer_not_auto_included_or_excluded() -> None:
    roles = _roles()
    thin = classify_role("Data Engineer", "Work with stakeholders on metrics.", roles)
    assert thin.needs_llm is True
    rich = classify_role(
        "Data Engineer",
        "Design and implement production pipelines, write code, and operate microservices and CI/CD.",
        roles,
    )
    assert rich.is_software_engineering is True


def test_required_years_over_cap_rejected() -> None:
    roles = _roles()
    verdict = classify_seniority(
        "Software Engineer",
        "Minimum Qualifications\n5+ years of professional software experience.",
        roles,
    )
    assert verdict.fits_entry_level is False
    assert verdict.needs_llm is False


def test_preferred_years_do_not_reject() -> None:
    roles = _roles()
    verdict = classify_seniority(
        "Software Engineer",
        "Minimum Qualifications\n0-2 years of experience.\nPreferred Qualifications\n5+ years of industry experience.",
        roles,
    )
    assert verdict.fits_entry_level is True
    parsed = extract_experience_requirement(
        "Minimum Qualifications\n0-2 years of experience.\nPreferred Qualifications\n5+ years of industry experience."
    )
    assert parsed.min_years == 0
    assert parsed.preferred_min_years == 5


def test_entry_level_signals() -> None:
    roles = _roles()
    for text in ("new graduates welcome", "entry-level", "early career", "university graduate"):
        verdict = classify_seniority("Software Engineer", text, roles)
        assert verdict.fits_entry_level is True, text


def test_no_experience_requirement_with_entry_signal_is_kept() -> None:
    roles = _roles()
    verdict = classify_seniority(
        "Software Engineer",
        "We welcome recent graduates. No prior industry experience required.",
        roles,
    )
    assert verdict.fits_entry_level is True


def test_required_two_years_fits() -> None:
    roles = _roles()
    verdict = classify_seniority(
        "Backend Engineer",
        "Required Qualifications\n1-2 years of software experience.",
        roles,
    )
    assert verdict.fits_entry_level is True


def test_senior_title_rejected() -> None:
    roles = _roles()
    for title in (
        "Senior Software Engineer",
        "Staff Engineer",
        "Staff Software Engineer",
        "Lead Software Engineer",
        "Principal Software Engineer",
        "Engineering Lead",
    ):
        verdict = classify_seniority(
            title,
            "Required Qualifications\n0-2 years of experience.",
            roles,
        )
        assert verdict.fits_entry_level is False, title


def test_numbered_title_with_required_zero_to_two_stays_eligible() -> None:
    roles = _roles()
    verdict = classify_seniority(
        "Software Engineer II",
        "Required Qualifications\n0-2 years of experience.",
        roles,
    )
    assert verdict.fits_entry_level is True


def test_us_and_international_locations() -> None:
    seattle = normalize_location("Seattle, WA")
    assert seattle.is_us is True
    london = normalize_location("London, United Kingdom")
    assert london.is_international_only is True
    remote_us = normalize_location("Remote - United States")
    assert remote_us.is_us is True
    assert remote_us.remote_type.value == "Remote"


def test_employment_types() -> None:
    assert detect_employment_type("Full-time").value == "Full-time"
    assert detect_employment_type("Internship").value == "Internship"
    assert detect_employment_type("Co-op").value == "Co-op"
    assert detect_employment_type("Contract").value == "Contract"
    assert detect_employment_type("Part-time").value == "Part-time"
