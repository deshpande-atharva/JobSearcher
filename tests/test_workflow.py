"""Contract for the unattended production workflow.

Parses the workflow that GitHub Actions will run. This does not execute a
hosted runner.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "daily_jobs.yml"

_OPTIONAL_SECRETS = (
    "GEMINI_API_KEY",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "NOTIFICATION_EMAIL",
)


def _load() -> dict:
    text = WORKFLOW.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def test_workflow_yaml_parses_and_schedules_production() -> None:
    data = _load()
    # PyYAML 1.1 reads the bare GitHub key `on` as boolean True.
    triggers = data[True]
    assert "workflow_dispatch" in triggers
    assert triggers["schedule"] == [{"cron": "0 12 * * *"}]
    assert list(data["permissions"]) == ["contents"]
    assert data["permissions"]["contents"] == "write"
    assert len(data["jobs"]) == 1

    job = data["jobs"]["discover"]
    assert job["runs-on"] == "ubuntu-latest"
    setup = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-python"))
    assert setup["with"]["python-version"] == "3.12"

    install = next(step for step in job["steps"] if "pip install -e ." in step.get("run", ""))
    assert "python -m venv .venv" in install["run"]

    browser = next(step for step in job["steps"] if "playwright" in step.get("run", ""))
    assert "python -m playwright install --with-deps chromium" in browser["run"]

    pipeline = next(step for step in job["steps"] if "python -m src.main" in step.get("run", ""))
    command = pipeline["run"]
    assert "python -m src.main" in command
    for flag in ("--fixture-mode", "--dry-run", "--diagnostic", "--freshness-hours"):
        assert flag not in command
    for name in _OPTIONAL_SECRETS:
        assert pipeline["env"][name] == f"${{{{ secrets.{name} }}}}"


def test_workflow_commits_only_tracker_paths() -> None:
    job = _load()["jobs"]["discover"]
    commit = next(step for step in job["steps"] if "git commit" in step.get("run", ""))
    script = commit["run"]
    assert "git add data/current data/archive" in script
    assert "git diff --cached --quiet" in script
    assert "git fetch origin main" in script
    assert "git rebase origin/main" in script
    assert "git push --force" not in script
    assert "git push origin HEAD:main" in script
    assert "git add -A" not in script
    assert "git add ." not in script
    assert ".env" not in script
    assert ".venv" not in script.split("git add", 1)[1]
