from src.models.config import CompanyConfig
from src.services.ats_discovery import detect_ats_from_html, detect_ats_from_url
from src.services.ats_registry import AtsRegistry, AtsRegistryEntry
from src.utils.normalization import normalize_company_name


def test_greenhouse_detected_from_url() -> None:
    hit = detect_ats_from_url("https://boards.greenhouse.io/example")
    assert hit is not None
    assert hit.ats_type == "greenhouse"
    assert hit.identifier == "example"


def test_lever_detected_from_url() -> None:
    hit = detect_ats_from_url("https://jobs.lever.co/northwind/abc")
    assert hit is not None
    assert hit.ats_type == "lever"
    assert hit.identifier == "northwind"


def test_ashby_detected_from_url() -> None:
    hit = detect_ats_from_url("https://jobs.ashbyhq.com/acme")
    assert hit is not None
    assert hit.ats_type == "ashby"
    assert hit.identifier == "acme"


def test_smartrecruiters_detected_from_url() -> None:
    hit = detect_ats_from_url("https://jobs.smartrecruiters.com/ExampleCompany")
    assert hit is not None
    assert hit.ats_type == "smartrecruiters"
    assert hit.identifier == "ExampleCompany"


def test_workday_detected_from_url() -> None:
    url = "https://elevancehealth.wd1.myworkdayjobs.com/en-US/ELV_EXT"
    hit = detect_ats_from_url(url)
    assert hit is not None
    assert hit.ats_type == "workday"
    assert "ELV_EXT" in hit.identifier


def test_icims_detected_from_url() -> None:
    url = "https://example.icims.com/jobs/1234/software-engineer/job"
    hit = detect_ats_from_url(url)
    assert hit is not None
    assert hit.ats_type == "icims"
    assert "icims.com" in hit.identifier


def test_no_ats_detected_on_generic_careers_url() -> None:
    assert detect_ats_from_url("https://example.com/careers") is None
    assert detect_ats_from_html("<html><body>Join our team</body></html>") is None


def test_html_greenhouse_embed() -> None:
    html = '<iframe src="https://boards.greenhouse.io/embed/job_board?for=stripe"></iframe>'
    hit = detect_ats_from_html(html)
    assert hit is not None
    assert hit.ats_type == "greenhouse"
    assert hit.identifier == "stripe"


def test_manual_override_beats_automatic_url() -> None:
    company = CompanyConfig(
        name="Example",
        careers_url="https://jobs.lever.co/otherco",
        ats_type="greenhouse",
        ats_identifier="example",
    )
    assert company.ats_discovery_mode == "manual"
    assert company.has_structured_discovery is True
    auto = detect_ats_from_url(company.careers_url)
    assert auto is not None and auto.ats_type == "lever"
    assert company.ats_type == "greenhouse"
    assert company.ats_identifier == "example"


def test_nested_manual_ats_block() -> None:
    company = CompanyConfig(
        name="Nested",
        careers_url="https://example.com/careers",
        ats={"type": "greenhouse", "identifier": "nestedco", "discovery": "manual"},
    )
    assert company.ats_type == "greenhouse"
    assert company.ats_identifier == "nestedco"


def test_registry_manual_not_required_for_cache(tmp_path) -> None:
    registry = AtsRegistry(tmp_path / "ats_registry.yaml")
    registry.remember(
        "Example Company",
        ats_type="greenhouse",
        ats_identifier="example",
        method="careers_url",
    )
    entry = registry.get("Example Company")
    assert isinstance(entry, AtsRegistryEntry)
    assert entry.ats_identifier == "example"
    assert registry.get("example company") is not None
    assert normalize_company_name("Example Company") in (
        normalize_company_name("Example Company"),
    )
