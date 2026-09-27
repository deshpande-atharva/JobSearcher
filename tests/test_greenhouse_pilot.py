"""Greenhouse pilot tests. No live network and no Playwright launch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import FitResult, evaluate_job
from src.agents.navigation_agent import NavigationChoice, decide_action
from src.browser.actions import ActionError, BrowserAction, validate_action
from src.browser.page_state import PageState
from src.browser.session import InMemoryBrowser, PageFixture
from src.models.job import RawJobPosting
from src.navigation.loop import run_navigation
from src.navigation.memory import NavigationMemory, StrategyRecord, StrategyStep, StrategyVersion
from src.pilot.greenhouse import cards_to_postings, discover_structured
from src.services.candidate_profile import CandidateProfile, load_candidate_profile
from src.sources.base import HttpClient, SourceContext
from src.utils.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "tests" / "fixtures" / "greenhouse" / "pages"


def _browser(start: str) -> InMemoryBrowser:
    pages: dict[str, PageFixture] = {}
    for path in PAGES.glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        transitions = raw.pop("transitions", {})
        page_id = raw["id"]
        pages[page_id] = PageFixture(id=page_id, state=PageState.model_validate(raw), transitions=transitions)
    return InMemoryBrowser(pages, start)


class ScriptedLLM:
    available = True

    def __init__(self, choice: NavigationChoice) -> None:
        self.choice = choice
        self.calls = 0

    async def structured(self, **kwargs: object) -> NavigationChoice:
        self.calls += 1
        return self.choice


def test_browser_click_type_select_scroll_back() -> None:
    browser = _browser("career_home")

    async def run() -> None:
        page = await browser.get_page_state()
        assert page.buttons[0].text == "View Jobs"
        await browser.execute(BrowserAction(action="type", target_id="in_search", value="python"))
        await browser.execute(BrowserAction(action="select", target_id="sel_dept", value="Engineering"))
        await browser.execute(BrowserAction(action="scroll"))
        await browser.execute(BrowserAction(action="click", target_id="btn_view", text="View Jobs"))
        listed = await browser.get_page_state()
        assert listed.job_cards[0].job_id == "1001"
        await browser.back()
        home = await browser.get_page_state()
        assert home.title == "Acme Careers"

    import asyncio

    asyncio.run(run())
    assert browser.typed == [("in_search", "python")]
    assert browser.selected == [("sel_dept", "Engineering")]
    assert browser.scrolls == 1


def test_invalid_script_action_is_rejected() -> None:
    page = PageState(buttons=[{"id": "btn_1", "text": "View Jobs", "role": "button"}])
    with pytest.raises(ActionError):
        validate_action(BrowserAction(action="click", target_id="btn_1", script="alert(1)"), page)


def test_submit_click_is_rejected() -> None:
    page = PageState(buttons=[{"id": "btn_1", "text": "Submit application", "role": "button"}])
    with pytest.raises(ActionError):
        validate_action(BrowserAction(action="click", target_id="btn_1"), page)


@pytest.mark.asyncio
async def test_known_strategy_is_reused(tmp_path: Path) -> None:
    memory = NavigationMemory(tmp_path)
    record = StrategyRecord(company="Acme", career_url="https://boards.greenhouse.io/acme")
    memory.add_version(
        record,
        [StrategyStep(action="click", semantic_target="view_jobs", text="View Jobs")],
        validated=True,
    )

    async def decide(page):
        return await decide_action(page, None)

    result = await run_navigation(
        _browser("career_home"),
        company="Acme",
        career_url=record.career_url,
        memory=memory,
        decide=decide,
        max_steps=5,
        max_seconds=5,
    )
    assert result.stopped == "strategy_reused"
    assert result.page.job_cards
    saved = memory.load("Acme")
    assert saved is not None
    assert saved.current is not None
    assert saved.current.success_count == 2
    assert saved.current.actions[0].text == "View Jobs"


@pytest.mark.asyncio
async def test_changed_wording_recovers_and_keeps_old_strategy(tmp_path: Path) -> None:
    memory = NavigationMemory(tmp_path)
    record = StrategyRecord(company="Acme", career_url="https://boards.greenhouse.io/acme")
    memory.add_version(
        record,
        [StrategyStep(action="click", semantic_target="view_jobs", text="View Jobs")],
        validated=True,
    )
    llm = ScriptedLLM(
        NavigationChoice(
            target_id="btn_explore",
            semantic_role="view_jobs",
            text="Explore Opportunities",
            reason="Explore Opportunities is the same control as View Jobs.",
        )
    )

    async def decide(page):
        return await decide_action(page, llm)

    result = await run_navigation(
        _browser("changed_layout"),
        company="Acme",
        career_url=record.career_url,
        memory=memory,
        decide=decide,
        max_steps=5,
        max_seconds=5,
    )
    assert result.stopped == "recovered"
    assert result.recovery is True
    assert llm.calls == 0
    assert result.page.job_cards[0].job_id == "1001"
    saved = memory.load("Acme")
    assert saved is not None
    assert saved.versions[0].actions[0].text == "View Jobs"
    assert saved.versions[0].failure_count == 1
    assert saved.versions[1].actions[0].text == "Explore Opportunities"
    assert saved.versions[1].status == "validated"
    assert saved.versions[1].version == 2


@pytest.mark.asyncio
async def test_navigation_stops_on_timeout_steps_and_repeats(tmp_path: Path) -> None:
    memory = NavigationMemory(tmp_path)

    async def no_action(page):
        return None

    timed = await run_navigation(
        _browser("pagination"),
        company="Acme",
        career_url="https://boards.greenhouse.io/acme",
        memory=memory,
        decide=no_action,
        max_steps=5,
        max_seconds=-1,
    )
    assert timed.stopped == "timeout"

    async def stuck(page):
        return BrowserAction(action="click", target_id="missing")

    invalid = await run_navigation(
        _browser("pagination"),
        company="Other",
        career_url="https://boards.greenhouse.io/acme",
        memory=NavigationMemory(tmp_path / "b"),
        decide=stuck,
        max_steps=3,
        max_seconds=5,
    )
    assert invalid.stopped == "invalid_action"

    async def same(page):
        return BrowserAction(action="scroll")

    repeated = await run_navigation(
        _browser("pagination"),
        company="Scroll",
        career_url="https://boards.greenhouse.io/acme",
        memory=NavigationMemory(tmp_path / "c"),
        decide=same,
        max_steps=4,
        max_seconds=5,
    )
    assert repeated.stopped == "repeated_action"


@pytest.mark.asyncio
async def test_load_more_and_job_extraction(tmp_path: Path) -> None:
    async def decide(page):
        return await decide_action(page, None)

    result = await run_navigation(
        _browser("pagination"),
        company="Acme",
        career_url="https://boards.greenhouse.io/acme",
        memory=NavigationMemory(tmp_path),
        decide=decide,
        max_steps=4,
        max_seconds=5,
    )
    assert result.page.job_cards
    postings = cards_to_postings(result.page, "Acme")
    assert {item.job_id for item in postings} >= {"1001", "1002"}
    assert all(item.apply_url and "greenhouse.io" in item.apply_url for item in postings)


@pytest.mark.asyncio
async def test_structured_greenhouse_fixture(tmp_config) -> None:
    company = tmp_config.universe.find("Acme Robotics")
    assert company is not None
    http = HttpClient(tmp_config)
    ctx = SourceContext(config=tmp_config, http=http, logger=get_logger("test"))
    jobs = await discover_structured(ctx, company)
    await http.aclose()
    assert {job.job_id for job in jobs} >= {"1001", "1002"}
    assert all(job.provenance.get("discovery_method") == "structured" for job in jobs)
    assert all("greenhouse.io" in (job.apply_url or "") for job in jobs)


def test_linkedin_url_is_not_used() -> None:
    page = PageState(
        url="https://example.test",
        job_cards=[
            {
                "id": "c",
                "title": "Software Engineer",
                "url": "https://www.linkedin.com/jobs/view/1",
                "job_id": "1",
            }
        ],
    )
    posting = cards_to_postings(page, "Acme")[0]
    assert posting.apply_url is None
    assert posting.provenance["url_status"] == "UNKNOWN"


def test_profile_does_not_invent_skills(project_root: Path) -> None:
    profile = load_candidate_profile(project_root / "config" / "candidate_profile.yaml")
    assert profile.technical_skills == []
    assert profile.experience_evidence == []
    assert profile.education == []
    assert profile.max_required_years == 2


def test_semantic_labels_match_without_exact_duplicates() -> None:
    from src.navigation.semantics import SEMANTIC_TARGETS, find_semantic
    from src.browser.page_state import PageState

    for text in ("View Jobs", "Explore Opportunities", "Open Positions", "Search Jobs"):
        page = PageState(buttons=[{"id": "b", "text": text, "role": "button"}])
        found = find_semantic(page, "view_jobs")
        assert found is not None and found.text == text
        assert text.lower() in SEMANTIC_TARGETS["view_jobs"]
    for text in ("Load More", "Show More", "Next", "Next Page", "Go to page 2"):
        page = PageState(links=[{"id": "a", "text": text, "role": "link"}])
        found = find_semantic(page, "paginate")
        assert found is not None and found.text == text


def test_candidate_strategy_is_promoted_only_after_validation(tmp_path: Path) -> None:
    memory = NavigationMemory(tmp_path)
    record = StrategyRecord(company="Acme", career_url="https://boards.greenhouse.io/acme")
    version = memory.add_version(
        record,
        [StrategyStep(action="click", semantic_target="view_jobs", text="View Jobs")],
        validated=False,
    )
    saved = memory.load("Acme")
    assert saved is not None
    assert saved.versions[0].status == "candidate"
    memory.promote(saved, saved.versions[0])
    promoted = memory.load("Acme")
    assert promoted is not None
    assert promoted.versions[0].status == "validated"
    assert promoted.versions[0].version == version.version


@pytest.mark.asyncio
async def test_unknown_label_uses_llm_recovery(tmp_path: Path) -> None:
    from src.browser.page_state import JobCard, PageState
    from src.browser.session import InMemoryBrowser, PageFixture

    home = PageState(
        url="https://boards.greenhouse.io/acme",
        title="Careers",
        buttons=[{"id": "btn_openings", "text": "See our openings", "role": "button"}],
    )
    listing = PageState(
        url="https://boards.greenhouse.io/acme/jobs",
        title="Jobs",
        job_cards=[
            JobCard(
                id="card",
                title="Software Engineer",
                url="https://boards.greenhouse.io/acme/jobs/1001",
                job_id="1001",
            )
        ],
    )
    browser = InMemoryBrowser(
        {
            "home": PageFixture(id="home", state=home, transitions={"btn_openings": "listing"}),
            "listing": PageFixture(id="listing", state=listing),
        },
        "home",
    )
    memory = NavigationMemory(tmp_path)
    record = StrategyRecord(company="Acme", career_url=home.url)
    memory.add_version(
        record,
        [StrategyStep(action="click", semantic_target="view_jobs", text="View Jobs")],
        validated=True,
    )
    llm = ScriptedLLM(
        NavigationChoice(
            target_id="btn_openings",
            semantic_role="view_jobs",
            text="See our openings",
            reason="This control opens the job list.",
        )
    )

    async def decide(page):
        return await decide_action(page, llm)

    result = await run_navigation(
        browser,
        company="Acme",
        career_url=home.url,
        memory=memory,
        decide=decide,
        max_steps=4,
        max_seconds=5,
    )
    assert llm.calls == 1
    assert result.stopped == "recovered"
    assert result.page.job_cards[0].job_id == "1001"
    saved = memory.load("Acme")
    assert saved is not None
    assert saved.versions[0].status == "validated"
    assert saved.versions[0].actions[0].text == "View Jobs"
    assert saved.versions[1].status == "validated"
    assert saved.versions[1].actions[0].text == "See our openings"


def test_required_years_reject_and_preferred_years_do_not(tmp_config) -> None:
    profile = CandidateProfile(technical_skills=["Python"])
    rejected = evaluate_job(
        RawJobPosting(
            source="greenhouse",
            company_name="Acme",
            title="Senior Software Engineer",
            description="5+ years required. Python.",
        ),
        profile,
        tmp_config.roles,
    )
    assert rejected.decision == "REJECT"
    kept = evaluate_job(
        RawJobPosting(
            source="greenhouse",
            company_name="Acme",
            title="Software Engineer",
            description="0-2 years of experience. Python. 2+ years preferred.",
        ),
        profile,
        tmp_config.roles,
    )
    assert kept.decision != "REJECT"
    assert "Python" in kept.matched_requirements


def test_critic_removes_unsupported_skill(tmp_config) -> None:
    posting = RawJobPosting(
        source="greenhouse",
        company_name="Acme",
        title="Software Engineer",
        description="Build APIs in Python.",
    )
    profile = CandidateProfile(technical_skills=["Python"])
    fit = FitResult(
        decision="STRONG_MATCH",
        confidence="high",
        role_alignment="strong",
        experience_alignment="strong",
        technical_alignment="strong",
        matched_requirements=["Python", "Kafka"],
        candidate_evidence=[
            {"requirement": "Python", "evidence": "listed"},
            {"requirement": "Kafka", "evidence": "invented"},
        ],
    )
    critique = review_fit(fit, posting, profile)
    assert "Kafka" in critique.unsupported_claims
    assert critique.fit.matched_requirements == ["Python"]
    assert all(item["requirement"] != "Kafka" for item in critique.fit.candidate_evidence)


def test_same_job_keeps_both_sources() -> None:
    from src.agents.discovery_agent import _dedupe_raw

    first = RawJobPosting(
        source="greenhouse",
        company_name="Acme",
        title="Software Engineer",
        job_id="1001",
        apply_url="https://boards.greenhouse.io/acme/jobs/1001",
    )
    second = RawJobPosting(
        source="jobright",
        company_name="Acme",
        title="Software Engineer",
        job_id="1001",
        apply_url="https://boards.greenhouse.io/acme/jobs/1001",
    )
    unique = _dedupe_raw([first, second])
    assert len(unique) == 1
    assert unique[0].provenance["discovered_from"] == ["greenhouse", "jobright"]


def test_same_title_different_ids_stay_separate() -> None:
    from src.services.deduplication import deduplicate
    from tests.conftest import make_job

    unique, duplicates, _ = deduplicate(
        [
            make_job(job_id="1001", job_title="Software Engineer"),
            make_job(job_id="1002", job_title="Software Engineer"),
        ]
    )
    assert len(unique) == 2
    assert duplicates == []
