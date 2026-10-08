"""Workday keyword-strategy memory, title policy, and the fixture qualification chain."""

from __future__ import annotations

from datetime import timedelta

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.browser.actions import ActionError, BrowserAction, validate_action
from src.browser.page_state import JobCard, PageElement, PageState
from src.browser.session import InMemoryBrowser, PageFixture
from src.main import build_parser
from src.models.job import DateSource, H1BLookupResult, LanguagePolarity, RawJobPosting, VisaSponsorshipStatus
from src.navigation.loop import _replay
from src.navigation.memory import NavigationMemory, StrategyRecord, StrategyStep
from src.pilot.workday import official_workday_url
from src.pilot.workday_qualify import qualify_workday_postings
from src.pilot.workday_target import (
    keyword_strategy_steps,
    search_state_valid,
    select_detail_candidates,
    strategy_submits_with_enter,
    worth_detail,
)
from src.services.candidate_profile import CandidateProfile, SkillEvidence
from src.services.freshness import is_fresh
from src.services.h1b import analyze_sponsorship, scan_sponsorship_language
from src.services.roles import classify_role
from src.services.seniority import classify_seniority
from src.services.url_verification import pick_direct_url, verify_url
from src.services.workday_detail import parse_workday_detail
from src.sources.base import FetchResult
from src.utils.dates import parse_labeled_job_date, utcnow
from tests.conftest import make_job

BOARD = "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite"
KEYWORD = "software engineer new grad"


def _url(job_id: str) -> str:
    return f"{BOARD}/job/Santa-Clara/Software-Engineer_{job_id}"


def _profile() -> CandidateProfile:
    return CandidateProfile(
        profile_version=1,
        professional_months=18,
        max_required_years=2,
        require_us_location=True,
        employment_types=["Full-time", "Contract", "Internship", "Co-op"],
        education=["Bachelors in Electronics and Telecommunication"],
        experience_evidence=[
            SkillEvidence(skill="Java", evidence="Developed backend services using Java.", status="EXPLICIT"),
            SkillEvidence(skill="PostgreSQL", evidence="Built REST APIs with PostgreSQL.", status="EXPLICIT"),
        ],
    )


def test_live_e2e_flag_is_opt_in() -> None:
    args = build_parser().parse_args(["--workday-live-e2e-smoke-test", "--no-email"])
    assert args.workday_live_e2e_smoke_test is True
    assert args.no_email is True


def test_enter_submit_is_allowed_only_on_a_text_input() -> None:
    page = PageState(
        inputs=[PageElement(id="in_1", text="Search for jobs or keywords", role="text")],
        buttons=[PageElement(id="btn_1", text="Submit application", role="button")],
    )
    action = validate_action(
        BrowserAction(action="submit", target_id="in_1", semantic_role="search_input", text="Search for jobs or keywords"),
        page,
    )
    assert action.action == "submit"
    try:
        validate_action(BrowserAction(action="submit", target_id="btn_1"), page)
    except ActionError:
        return
    raise AssertionError("submit on a button must be rejected")


def test_keyword_strategy_is_versioned_without_removing_the_old_one(tmp_path) -> None:
    memory = NavigationMemory(tmp_path)
    record = StrategyRecord(
        source="workday",
        source_type="workday",
        company="NVIDIA",
        career_url=BOARD,
    )
    memory.add_version(
        record,
        [StrategyStep(action="click", semantic_target="search_submit", text="Search")],
        validated=True,
    )
    memory.add_version(record, keyword_strategy_steps("Search for jobs or keywords", KEYWORD), validated=True)
    memory.add_version(
        record,
        [StrategyStep(action="click", semantic_target="search_submit", text="Search again")],
        validated=False,
    )
    loaded = memory.load("NVIDIA", "workday")
    assert loaded is not None
    assert [item.version for item in loaded.versions] == [1, 2, 3]
    assert loaded.versions[0].actions[0].action == "click"
    assert loaded.current is not None
    assert loaded.current.version == 2
    assert loaded.current.status == "validated"
    assert loaded.current.created_at
    assert strategy_submits_with_enter(loaded.current) is True
    assert strategy_submits_with_enter(loaded.versions[0]) is False


async def test_recorded_enter_strategy_reaches_filtered_results() -> None:
    home = PageFixture(
        id="home",
        state=PageState(
            url=BOARD,
            inputs=[PageElement(id="in_1", text="Search for jobs or keywords", role="text")],
            job_cards=[JobCard(id="c1", title="Senior Software Engineer", url=_url("JR1"), job_id="JR1")],
        ),
        transitions={"in_1": "results"},
    )
    results = PageFixture(
        id="results",
        state=PageState(
            url=f"{BOARD}?q=software%20engineer%20new%20grad",
            inputs=[PageElement(id="in_1", text="Search for jobs or keywords", role="text", value=KEYWORD)],
            job_cards=[
                JobCard(
                    id="c2",
                    title="Software Engineer - New College Grad",
                    url=_url("JR2"),
                    job_id="JR2",
                )
            ],
        ),
    )
    browser = InMemoryBrowser({"home": home, "results": results}, "home")
    before = {card.job_id for card in home.state.job_cards}
    _ok, page = await _replay(browser, keyword_strategy_steps("Search for jobs or keywords", KEYWORD))
    after = {card.job_id for card in page.job_cards}
    assert browser.submitted == ["in_1"]
    assert search_state_valid(page.url, KEYWORD, before, after) is True
    assert search_state_valid(BOARD, KEYWORD, before, before) is False


def test_title_policy_inspects_senior_software_and_excludes_management(tmp_config) -> None:
    roles = tmp_config.roles
    assert worth_detail("Senior Software Engineer", roles) is True
    assert worth_detail("Senior Backend Engineer", roles) is True
    assert worth_detail("Software Engineer II", roles) is True
    assert worth_detail("Engineering Manager", roles) is False
    assert worth_detail("Software Engineering Manager", roles) is False
    assert worth_detail("Director of Engineering", roles) is False
    assert classify_role(
        "Engineering Manager",
        "Minimum Qualifications\n0-2 years of experience.",
        roles,
    ).is_software_engineering is False
    assert worth_detail("Principal Software Engineer", roles) is False
    assert worth_detail("Principal Architect", roles) is False
    assert worth_detail("Enterprise Architect", roles) is False
    assert worth_detail("Staff Software Engineer", roles) is False

    early = [
        RawJobPosting(
            source="workday",
            company_name="NVIDIA",
            title=f"New Grad Software Engineer {index}",
            job_id=f"JR{index:02d}",
            apply_url=_url(f"JR{index:02d}"),
        )
        for index in range(15)
    ]
    seniors = [
        RawJobPosting(
            source="workday",
            company_name="NVIDIA",
            title=f"Senior Software Engineer {index}",
            job_id=f"JS{index:02d}",
            apply_url=_url(f"JS{index:02d}"),
        )
        for index in range(10)
    ]
    chosen, _skipped, unopened = select_detail_candidates([*seniors, *early], roles, limit=12)
    assert len(chosen) == 12
    assert all("New Grad" in posting.title for posting in chosen)
    assert unopened > 0


def test_experience_filter_uses_required_years(tmp_config) -> None:
    roles = tmp_config.roles
    senior_early = classify_seniority(
        "Senior Software Engineer",
        "Minimum Qualifications\n0-2 years of experience.",
        roles,
    )
    senior_five = classify_seniority(
        "Senior Software Engineer",
        "Minimum Qualifications\n5+ years of experience.",
        roles,
    )
    second_early = classify_seniority(
        "Software Engineer II",
        "Minimum Qualifications\n0-2 years of experience.",
        roles,
    )
    second_four = classify_seniority(
        "Software Engineer II",
        "Minimum Qualifications\n4+ years of experience.",
        roles,
    )
    assert senior_early.fits_entry_level is True
    assert senior_five.fits_entry_level is False
    assert second_early.fits_entry_level is True
    assert second_four.fits_entry_level is False


def test_labeled_dates_and_discovery_time(tmp_config) -> None:
    now = utcnow()
    yesterday = parse_labeled_job_date("posted on Posted Yesterday", now=now)
    assert yesterday is not None
    assert abs((now - yesterday).total_seconds() - 24 * 3600) < 5
    explicit = parse_labeled_job_date("Posted on September 25, 2026", now=now)
    assert explicit is not None and explicit.year == 2026
    updated = parse_labeled_job_date("Updated on September 26, 2026", now=now)
    assert updated is not None and updated.day == 26
    thirty_plus = parse_labeled_job_date("Posted 30+ Days Ago", now=now)
    assert thirty_plus is not None
    assert (now - thirty_plus).days >= 29
    detail = parse_workday_detail(
        "<p>Build services.</p>",
        posted_text="Updated on September 26, 2026",
    )
    assert detail.date_source is DateSource.UPDATED_DATE
    unknown = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Software Engineer",
        apply_url=_url("JR1"),
        date_source=DateSource.UNKNOWN,
    )
    fresh, _age = is_fresh(unknown, tmp_config.freshness_hours)
    assert fresh is False
    posted = unknown.model_copy(
        update={"posted_at": now - timedelta(hours=2), "date_source": DateSource.POSTED_DATE}
    )
    assert is_fresh(posted, tmp_config.freshness_hours)[0] is True
    old = unknown.model_copy(
        update={"posted_at": now - timedelta(hours=72), "date_source": DateSource.POSTED_DATE}
    )
    assert is_fresh(old, tmp_config.freshness_hours)[0] is False
    revised = unknown.model_copy(
        update={"updated_at": now - timedelta(hours=3), "date_source": DateSource.UPDATED_DATE}
    )
    assert is_fresh(revised, tmp_config.freshness_hours, use_updated_when_posted_missing=True)[0] is True


async def test_url_verification_keeps_official_workday_links(tmp_config) -> None:
    live = tmp_config.model_copy(update={"fixture_mode": False})
    posting = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Software Engineer",
        job_id="JR3001",
        apply_url=_url("JR3001"),
    )
    official, status = official_workday_url(posting.apply_url)
    assert status == "verified" and official
    blocked, blocked_status = official_workday_url("https://jobright.ai/jobs/1")
    assert blocked is None and blocked_status == "UNKNOWN"
    check = pick_direct_url(posting, live.url_policy)
    missed = await verify_url(check, live, _Http(404))
    # A failed live probe rejects the posting even when the host is Workday.
    assert missed.accepted is False
    assert missed.reachable is False
    assert missed.url == posting.apply_url
    other = RawJobPosting(
        source="company",
        company_name="Example",
        title="Software Engineer",
        apply_url="https://example.com/jobs/1",
    )
    foreign = pick_direct_url(other, live.url_policy)
    strict_urls = live.settings.urls.model_copy(update={"allow_unverified_ats_urls": False})
    strict = live.model_copy(update={"settings": live.settings.model_copy(update={"urls": strict_urls})})
    rejected = await verify_url(foreign, strict, _Http(404))
    assert rejected.accepted is False
    assert rejected.reachable is False


def test_sponsorship_language_is_not_inferred(tmp_config) -> None:
    positive = scan_sponsorship_language("H-1B sponsorship is available.")
    negative = scan_sponsorship_language("We will not sponsor work visas.")
    absent = scan_sponsorship_language("Java required.")
    assert positive.polarity is LanguagePolarity.POSITIVE
    assert negative.polarity is LanguagePolarity.NEGATIVE
    assert absent.polarity is LanguagePolarity.ABSENT
    assert scan_sponsorship_language("Sponsorship may be considered.").polarity is LanguagePolarity.AMBIGUOUS
    job = make_job()
    confirmed = analyze_sponsorship(job, positive, None, config=tmp_config)
    declined = analyze_sponsorship(job, negative, None, config=tmp_config)
    unknown = analyze_sponsorship(
        job,
        absent,
        H1BLookupResult(company=job.company, found=False, error="H1BGrader unavailable"),
        config=tmp_config,
    )
    assert confirmed.status is VisaSponsorshipStatus.CONFIRMED
    assert declined.status is VisaSponsorshipStatus.NOT_SUPPORTED
    assert unknown.status is VisaSponsorshipStatus.UNKNOWN
    assert unknown.lookup_failed is True


async def test_fixture_chain_scores_a_fresh_us_early_career_role(tmp_config) -> None:
    now = utcnow()
    posting = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Software Engineer",
        job_id="JR4100",
        apply_url=_url("JR4100"),
        location_raw="Santa Clara, CA",
        employment_type_raw="Full-time",
        posted_at=now - timedelta(hours=2),
        date_source=DateSource.POSTED_DATE,
        description=(
            "Minimum Qualifications\n"
            "0-2 years of experience.\n"
            "BS in Computer Science or Engineering.\n"
            "Java and PostgreSQL required.\n"
            "Kubernetes required.\n"
            "Preferred Qualifications\n"
            "Kafka is a plus.\n"
        ),
        provenance={"discovery_method": "browser"},
    )
    funnel = await qualify_workday_postings(tmp_config, [posting])
    assert [job.job_id for job in funnel.state.jobs] == ["JR4100"]
    profile = _profile()
    fit = evaluate_job(posting, profile, tmp_config.roles)
    critique = review_fit(fit, posting, profile)
    assert fit.experience_alignment == "strong"
    assert fit.education_alignment == "related"
    assert fit.location_alignment == "aligned"
    assert fit.freshness == "fresh"
    assert fit.sponsorship == LanguagePolarity.ABSENT.value
    assert "Java" in critique.fit.matched_requirements
    assert "PostgreSQL" in critique.fit.matched_requirements
    assert "Kubernetes" in critique.fit.missing_requirements
    assert "Kafka" not in critique.fit.matched_requirements
    assert critique.unsupported_claims == []
    identity = review_fit(fit.model_copy(update={"job_id": "JR9999"}), posting, profile)
    assert "job id does not match the posting" in identity.issues
    invented = fit.model_copy(deep=True)
    invented.matched_requirements = ["Java", "Kubernetes"]
    invented.candidate_evidence.append(
        {"requirement": "Kubernetes", "evidence": "Invented Kubernetes experience", "match": "MATCHED"}
    )
    invented.experience_alignment = "strong"
    invented.education_alignment = "equivalent"
    invented.sponsorship = "POSITIVE"
    invented.freshness = "fresh"
    cleaned = review_fit(invented, posting.model_copy(update={"posted_at": None, "date_source": DateSource.UNKNOWN}), profile)
    assert "Kubernetes" in cleaned.unsupported_claims
    assert cleaned.fit.education_alignment == "related"
    assert cleaned.fit.sponsorship == LanguagePolarity.ABSENT.value
    assert cleaned.fit.freshness == "unknown"
    overstated = fit.model_copy(update={"experience_alignment": "strong", "decision": "STRONG_MATCH"})
    senior = posting.model_copy(
        update={
            "title": "Senior Software Engineer",
            "description": "Minimum Qualifications\n5+ years of experience.\nJava required.\n",
        }
    )
    experience = review_fit(overstated, senior, profile)
    assert experience.fit.experience_alignment == "reject"


class _Http:
    def __init__(self, status: int) -> None:
        self.status = status

    async def head(self, url: str) -> FetchResult:
        return FetchResult(url=url, status=self.status)
