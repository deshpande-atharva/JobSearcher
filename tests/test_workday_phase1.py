"""Workday Phase 1 discovery tests. No live network and no Playwright launch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.navigation_agent import NavigationChoice
from src.browser.actions import ActionError, BrowserAction, validate_action
from src.browser.page_state import JobCard, PageState
from src.browser.session import InMemoryBrowser, PageFixture
from src.models.job import DateSource
from src.navigation.loop import run_navigation
from src.navigation.memory import NavigationMemory, StrategyRecord, StrategyStep
from src.navigation.semantics import find_semantic
from src.pilot.workday import browser_health, canonical_jobs, cards_to_postings, extract_workday_job_id, official_workday_url
from src.pilot.workday_browser import decide_workday
from src.services.workday_detect import detect_workday_tenant
from src.utils.urls import extract_job_id

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "tests" / "fixtures" / "workday" / "pages"
HTML = ROOT / "tests" / "fixtures" / "workday" / "html"

NVIDIA = "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite"
JOB_URL = f"{NVIDIA}/job/Santa-Clara/Software-Engineer_JR1001"


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


def test_workday_search_label_is_a_semantic_target() -> None:
    page = PageState(inputs=[{"id": "in_1", "text": "Search for jobs or keywords", "role": "textbox"}])
    found = find_semantic(page, "search_input")
    assert found is not None
    assert found.id == "in_1"


def test_workday_detected_from_public_url() -> None:
    hit = detect_workday_tenant(NVIDIA, company="NVIDIA")
    assert hit.ok
    assert hit.source_type == "WORKDAY"
    assert hit.tenant == "nvidia"
    assert hit.site == "NVIDIAExternalCareerSite"
    assert hit.detection_method == "url"
    assert hit.host.endswith("myworkdayjobs.com")


def test_workday_detected_from_landing_html() -> None:
    html = (HTML / "landing.html").read_text(encoding="utf-8")
    hit = detect_workday_tenant("https://www.nvidia.com/careers", html, company="NVIDIA")
    assert hit.ok
    assert hit.tenant == "nvidia"
    assert hit.detection_method in {"html_embed", "html_cxs", "html_script"}


def test_workday_detected_from_escaped_html() -> None:
    html = (HTML / "escaped.html").read_text(encoding="utf-8")
    hit = detect_workday_tenant(None, html, company="NVIDIA")
    assert hit.ok
    assert hit.tenant == "nvidia"


def test_non_workday_page_is_not_detected() -> None:
    hit = detect_workday_tenant("https://boards.greenhouse.io/airbnb", "<html>jobs</html>")
    assert hit.ok is False


def test_job_id_and_official_url_from_workday_link() -> None:
    assert extract_job_id(JOB_URL) == "JR1001"
    assert extract_workday_job_id(JOB_URL) == "JR1001"
    url, status = official_workday_url(JOB_URL)
    assert url == JOB_URL
    assert status == "verified"
    rejected, _ = official_workday_url("https://www.linkedin.com/jobs/view/1")
    assert rejected is None


def test_cards_become_canonical_jobs_with_browser_provenance() -> None:
    page = PageState(
        url=NVIDIA,
        job_cards=[
            JobCard(
                id="c1",
                title="Software Engineer",
                company="NVIDIA",
                location="Santa Clara, CA",
                url=JOB_URL,
                job_id="JR1001",
            ),
            JobCard(
                id="c2",
                title="Software Engineer",
                company="NVIDIA",
                location="Santa Clara, CA",
                url=JOB_URL,
                job_id="JR1001",
            ),
        ],
    )
    postings = cards_to_postings(page, "NVIDIA")
    assert postings[0].source == "workday"
    assert postings[0].provenance["discovery_method"] == "browser"
    assert postings[0].date_source is DateSource.UNKNOWN
    jobs = canonical_jobs(postings)
    assert len(jobs) == 1
    assert jobs[0].source == "workday"
    assert jobs[0].job_id == "JR1001"
    assert jobs[0].direct_application_url == JOB_URL
    assert jobs[0].company == "NVIDIA"


@pytest.mark.asyncio
async def test_search_filter_and_pagination_from_fixtures() -> None:
    browser = _browser("career_home")
    page = await browser.get_page_state()
    search = find_semantic(page, "search_input")
    assert search is not None
    page = await browser.execute(
        BrowserAction(action="type", target_id=search.id, text=search.text, value="Software Engineer")
    )
    assert browser.typed == [("in_search", "Software Engineer")]
    location = find_semantic(page, "location_filter")
    assert location is not None
    submit = find_semantic(page, "search_submit")
    assert submit is not None
    page = await browser.execute(BrowserAction(action="click", target_id=submit.id, text=submit.text))
    assert [card.job_id for card in page.job_cards][:2] == ["JR1001", "JR1002"]
    pager = find_semantic(page, "paginate")
    assert pager is not None
    page = await browser.execute(BrowserAction(action="click", target_id=pager.id, text=pager.text))
    assert any(card.job_id == "JR1003" for card in page.job_cards)


@pytest.mark.asyncio
async def test_strategy_is_persisted_and_reused(tmp_path: Path) -> None:
    memory = NavigationMemory(tmp_path)
    record = StrategyRecord(
        source="workday",
        source_type="workday",
        company="NVIDIA",
        career_url=NVIDIA,
    )
    memory.add_version(
        record,
        [
            StrategyStep(action="type", semantic_target="search_input", text="Search for jobs", value="Software Engineer"),
            StrategyStep(action="click", semantic_target="search_submit", text="Search Jobs"),
        ],
        validated=True,
    )
    saved = tmp_path / "workday-nvidia.json"
    assert saved.exists()
    greenhouse = tmp_path / "greenhouse-nvidia.json"
    assert not greenhouse.exists()

    async def decide(page):
        return await decide_workday(page, None)

    result = await run_navigation(
        _browser("career_home"),
        company="NVIDIA",
        career_url=NVIDIA,
        memory=memory,
        decide=decide,
        max_steps=5,
        max_seconds=5,
        source="workday",
    )
    assert result.stopped == "strategy_reused"
    assert result.page.job_cards
    loaded = memory.load("NVIDIA", source="workday")
    assert loaded is not None
    assert loaded.current is not None
    assert loaded.current.success_count == 2


@pytest.mark.asyncio
async def test_changed_wording_recovers_without_replacing_old_strategy(tmp_path: Path) -> None:
    memory = NavigationMemory(tmp_path)
    record = StrategyRecord(source="workday", source_type="workday", company="NVIDIA", career_url=NVIDIA)
    memory.add_version(
        record,
        [StrategyStep(action="click", semantic_target="view_jobs", text="Search Jobs")],
        validated=True,
    )
    llm = ScriptedLLM(
        NavigationChoice(target_id="btn_explore", semantic_role="view_jobs", text="Search Openings", reason="same control")
    )

    async def decide(page):
        return await decide_workday(page, llm)

    result = await run_navigation(
        _browser("changed_layout"),
        company="NVIDIA",
        career_url=NVIDIA,
        memory=memory,
        decide=decide,
        max_steps=5,
        max_seconds=5,
        source="workday",
    )
    assert result.stopped == "recovered"
    assert result.recovery is True
    assert llm.calls == 0
    assert result.page.job_cards[0].job_id == "JR1001"
    saved = memory.load("NVIDIA", source="workday")
    assert saved is not None
    assert saved.versions[0].actions[0].text == "Search Jobs"
    assert saved.versions[0].failure_count == 1
    assert saved.versions[1].actions[0].text == "Search Openings"
    assert saved.versions[1].status == "validated"


def test_zero_jobs_is_not_successful_discovery() -> None:
    assert browser_health(jobs=0, navigated=True) == "EMPTY"
    assert browser_health(jobs=3, navigated=True) == "OK"
    assert browser_health(blocked="captcha", jobs=0, navigated=True) == "BLOCKED"
    assert browser_health(jobs=0, navigated=False) == "ERROR"
    assert browser_health(jobs=1, navigated=True, parser_ok=False) == "ERROR"


def test_unknown_date_is_not_invented() -> None:
    page = PageState(
        url=NVIDIA,
        job_cards=[
            JobCard(id="c1", title="Software Engineer", company="NVIDIA", url=JOB_URL, job_id="JR1001")
        ],
    )
    posting = cards_to_postings(page, "NVIDIA")[0]
    assert posting.posted_at is None
    assert posting.date_source is DateSource.UNKNOWN
    assert posting.provenance["discovery_method"] == "browser"


def test_script_and_apply_actions_are_rejected() -> None:
    page = PageState(buttons=[{"id": "btn_1", "text": "Apply Now", "role": "button"}])
    with pytest.raises(ActionError):
        validate_action(BrowserAction(action="click", target_id="btn_1", script="alert(1)"), page)
    with pytest.raises(ActionError):
        validate_action(BrowserAction(action="click", target_id="btn_1"), page)
