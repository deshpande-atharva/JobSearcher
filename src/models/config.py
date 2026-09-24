"""Configuration models and loader.

Every YAML file under ``config/`` is validated into these models at startup, so
a typo fails immediately with a readable message instead of producing subtly
wrong behaviour twelve steps into the pipeline.

Secrets are read from the environment only -- never from the YAML files, and
never committed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.job import ACCEPTED_EMPLOYMENT_TYPES, EmploymentType
from src.utils.urls import UrlPolicy

__all__ = [
    "AppConfig",
    "AtsBlock",
    "CompanyConfig",
    "CompanyDiscoveryBlock",
    "ConfigError",
    "FilterSettings",
    "LLMSettings",
    "NotificationSettings",
    "OutputSettings",
    "RolesConfig",
    "RunSettings",
    "Secrets",
    "Settings",
    "VisaSettings",
    "load_config",
]

AtsType = Literal["greenhouse", "lever", "ashby", "smartrecruiters", "workday", "icims"]
AtsTypeOrAuto = Literal["greenhouse", "lever", "ashby", "smartrecruiters", "workday", "icims", "auto"]
DEFAULT_COMPANY_SOURCES: tuple[str, ...] = ("company_ats", "company_careers", "jobright")


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# settings.yaml
# ---------------------------------------------------------------------------


class RunSettings(_Base):
    freshness_hours: float = Field(default=24.0, gt=0)
    # When posted_at is missing, age may be taken from updated_at. The two
    # fields are never mixed into one fabricated posted date.
    freshness_use_updated_when_posted_missing: bool = True
    max_concurrency: int = Field(default=8, ge=1, le=64)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=1.5, gt=0)
    user_agent: str = "job-agent/1.0"
    # 0 disables the cap. 15000 covers the 35-company universe once Workday
    # CXS is paginating (~3100 Greenhouse + ~6300 Workday + other ATS).
    max_jobs_per_run: int = Field(default=15000, ge=0)


class JobrightSourceSettings(_Base):
    enabled: bool = True
    entry_url: str = "https://jobright.ai/entry-level-jobs"
    max_pages: int = Field(default=3, ge=1, le=25)
    allow_browser_render: bool = True


class SimpleSourceSettings(_Base):
    enabled: bool = True


class WorkdaySourceSettings(_Base):
    """Workday CXS settings.

    Public Workday tenants tested here reject ``limit`` above 20 with HTTP 400.
    ``max_concurrency`` bounds Workday only; other ATS adapters keep the global
    run concurrency.
    """

    enabled: bool = True
    page_size: int = Field(default=20, ge=1, le=20)
    max_jobs: int = Field(default=2000, ge=20, le=5000)
    max_concurrency: int = Field(default=2, ge=1, le=8)


class DiscoverySources(_Base):
    jobright: JobrightSourceSettings = JobrightSourceSettings()
    greenhouse: SimpleSourceSettings = SimpleSourceSettings()
    lever: SimpleSourceSettings = SimpleSourceSettings()
    ashby: SimpleSourceSettings = SimpleSourceSettings()
    smartrecruiters: SimpleSourceSettings = SimpleSourceSettings()
    workday: WorkdaySourceSettings = WorkdaySourceSettings()
    icims: SimpleSourceSettings = SimpleSourceSettings()
    company_career: SimpleSourceSettings = SimpleSourceSettings()

    def is_enabled(self, name: str) -> bool:
        entry = getattr(self, name, None)
        return bool(entry and entry.enabled)


class DiscoverySettings(_Base):
    sources: DiscoverySources = DiscoverySources()
    ats_registry: str = "config/ats_registry.yaml"
    auto_detect_ats: bool = True
    persist_ats_registry: bool = True
    career_detail_fetch_limit: int = Field(default=12, ge=0, le=80)
    skip_career_page_when_ats_has_jobs: bool = True


class FilterSettings(_Base):
    employment_types: tuple[EmploymentType, ...] = tuple(sorted(ACCEPTED_EMPLOYMENT_TYPES))
    max_required_years: float = Field(default=2.0, ge=0)
    ignore_preferred_experience: bool = True
    require_us_location: bool = True
    reject_seniority_terms: tuple[str, ...] = ()

    @field_validator("employment_types", mode="before")
    @classmethod
    def _coerce_employment_types(cls, value: Any) -> Any:
        if value is None:
            return tuple(sorted(ACCEPTED_EMPLOYMENT_TYPES))
        if isinstance(value, str):
            value = [value]
        resolved: list[EmploymentType] = []
        for item in value:
            if isinstance(item, EmploymentType):
                resolved.append(item)
                continue
            key = str(item).strip().lower().replace("_", "-").replace(" ", "-")
            match = next(
                (et for et in EmploymentType if et.value.lower().replace(" ", "-") == key),
                None,
            )
            if match is None:
                raise ValueError(
                    f"unknown employment type {item!r}; expected one of "
                    f"{[et.value for et in EmploymentType]}"
                )
            resolved.append(match)
        return tuple(resolved)


class UrlSettings(_Base):
    aggregator_hosts: tuple[str, ...] = ()
    ats_hosts: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    strip_query_params: tuple[str, ...] = ()
    verify_reachability: bool = True
    allow_unverified_ats_urls: bool = True

    def to_policy(self) -> UrlPolicy:
        return UrlPolicy(
            aggregator_hosts=self.aggregator_hosts,
            ats_hosts=dict(self.ats_hosts),
            strip_query_params=self.strip_query_params,
        )


class VisaSettings(_Base):
    """H-1B evidence settings.

    Note what is absent: there is no option to require sponsorship. Sponsorship
    evidence is enrichment, and the pipeline has no code path that lets it
    remove a job. Unknown keys are rejected (``extra="forbid"``), so adding
    ``require_h1b_sponsorship: true`` to the YAML fails loudly at startup rather
    than quietly filtering jobs.
    """

    enabled: bool = True
    provider: Literal["h1bgrader", "none"] = "h1bgrader"
    base_url: str = "https://h1bgrader.com/h1b-sponsors"
    historical_lookback_years: int = Field(default=5, ge=1, le=20)
    use_job_title_matching: bool = True
    use_location_matching: bool = True
    use_current_posting_language: bool = True
    use_gemini_for_ambiguous_language: bool = True
    cache_ttl_hours: float = Field(default=168.0, ge=0)
    cache_path: str = "data/cache/h1b_lookups.json"
    skip_lookup_when_job_explicit: bool = True
    explicit_language_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class LLMSettings(_Base):
    provider: Literal["gemini", "none"] = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, ge=64)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_calls_per_run: int = Field(default=300, ge=0)
    circuit_breaker_failures: int = Field(default=5, ge=1)
    retry_attempts: int = Field(default=2, ge=0, le=5)


class OutputSettings(_Base):
    current_workbook: str = "data/current/jobs.xlsx"
    archive_dir: str = "data/archive"
    history_lookback_days: int = Field(default=120, ge=0)
    write_archive: bool = True


class NotificationSettings(_Base):
    enabled: bool = True
    subject_prefix: str = "[job-agent]"
    max_sample_jobs: int = Field(default=15, ge=0, le=200)


class PlaywrightSettings(_Base):
    enabled: bool = True
    headless: bool = True
    timeout_seconds: float = Field(default=45.0, gt=0)


class ScrapingSettings(_Base):
    respect_robots_txt: bool = True
    min_delay_seconds: float = Field(default=1.0, ge=0)
    max_html_bytes: int = Field(default=4_000_000, ge=1000)
    playwright: PlaywrightSettings = PlaywrightSettings()


class FixtureSettings(_Base):
    directory: str = "tests/fixtures"


class Settings(_Base):
    """Root model for ``config/settings.yaml``."""

    run: RunSettings = RunSettings()
    discovery: DiscoverySettings = DiscoverySettings()
    filters: FilterSettings = FilterSettings()
    urls: UrlSettings = UrlSettings()
    visa: VisaSettings = VisaSettings()
    llm: LLMSettings = LLMSettings()
    output: OutputSettings = OutputSettings()
    notifications: NotificationSettings = NotificationSettings()
    scraping: ScrapingSettings = ScrapingSettings()
    fixtures: FixtureSettings = FixtureSettings()


# ---------------------------------------------------------------------------
# roles.yaml
# ---------------------------------------------------------------------------


class RoleFamily(_Base):
    label: str
    keywords: tuple[str, ...] = ()


class SenioritySettings(_Base):
    entry_level_signals: tuple[str, ...] = ()
    reject_title_signals: tuple[str, ...] = ()
    max_required_years: float = Field(default=2.0, ge=0)


class RolesConfig(_Base):
    """Root model for ``config/roles.yaml``."""

    role_families: dict[str, RoleFamily] = Field(default_factory=dict)
    core_families: tuple[str, ...] = ()
    ambiguous_families: tuple[str, ...] = ()
    excluded_title_terms: tuple[str, ...] = ()
    software_work_signals: tuple[str, ...] = ()
    technology_signals: tuple[str, ...] = ()
    seniority: SenioritySettings = SenioritySettings()
    h1b_role_groups: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_family_references(self) -> RolesConfig:
        known = set(self.role_families)
        for bucket_name, bucket in (
            ("core_families", self.core_families),
            ("ambiguous_families", self.ambiguous_families),
        ):
            unknown = [name for name in bucket if name not in known]
            if unknown:
                raise ValueError(f"{bucket_name} references undefined role families: {unknown}")
        for group, families in self.h1b_role_groups.items():
            unknown = [name for name in families if name not in known]
            if unknown:
                raise ValueError(
                    f"h1b_role_groups[{group}] references undefined role families: {unknown}"
                )
        return self

    def label_for(self, family: str) -> str:
        entry = self.role_families.get(family)
        return entry.label if entry else family.replace("_", " ").title()

    def group_for(self, family: str) -> str | None:
        """H-1B comparison group a role family belongs to."""
        for group, families in self.h1b_role_groups.items():
            if family in families:
                return group
        return None


# ---------------------------------------------------------------------------
# companies.yaml
# ---------------------------------------------------------------------------


class AtsBlock(BaseModel):
    """Optional nested ATS block. Manual values always beat auto-detection."""

    model_config = ConfigDict(extra="ignore")

    type: AtsTypeOrAuto | None = None
    identifier: str | None = None
    discovery: Literal["manual", "auto"] = "auto"

    @field_validator("type", "identifier", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        return str(value).strip().lower()


class CompanyDiscoveryBlock(BaseModel):
    """Which discovery paths to attempt for this company."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    sources: tuple[str, ...] = DEFAULT_COMPANY_SOURCES


class CompanyConfig(BaseModel):
    """One entry from ``config/companies.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    aliases: tuple[str, ...] = ()
    fortune_500: bool = False
    careers_url: str | None = None
    ats_type: AtsType | None = None
    ats_identifier: str | None = None
    ats: AtsBlock | None = None
    discovery: CompanyDiscoveryBlock = Field(default_factory=CompanyDiscoveryBlock)
    discovery_urls: tuple[str, ...] = ()
    enabled: bool = True
    sponsorship_note: str | None = None

    @field_validator("ats_type", mode="before")
    @classmethod
    def _normalize_ats_type(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        return str(value).strip().lower()

    @field_validator("ats_identifier", "careers_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _sync_ats_block(self) -> CompanyConfig:
        """Copy a manual nested ``ats`` block onto the legacy fields."""
        block = self.ats
        if block and block.discovery == "manual" and block.type and block.type != "auto":
            if not block.identifier:
                raise ValueError(
                    f"company {self.name!r} sets ats.discovery=manual and ats.type="
                    f"{block.type!r} but no ats.identifier"
                )
            if not self.ats_type:
                self.ats_type = block.type  # type: ignore[assignment]
            if not self.ats_identifier:
                self.ats_identifier = block.identifier
        if self.ats_type and self.ats_type != "auto" and not self.ats_identifier:
            raise ValueError(
                f"company {self.name!r} sets ats_type={self.ats_type!r} but no ats_identifier; "
                "either add the verified identifier, use ats.discovery=auto, or leave ats_type null"
            )
        return self

    @property
    def has_structured_discovery(self) -> bool:
        return bool(self.ats_type and self.ats_identifier)

    @property
    def ats_discovery_mode(self) -> Literal["manual", "auto"]:
        if self.ats and self.ats.discovery == "manual":
            return "manual"
        if self.ats_type and self.ats_identifier:
            return "manual"
        return "auto"

    def wants_source(self, name: str) -> bool:
        if not self.discovery.enabled:
            return False
        return name in self.discovery.sources

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


class CompanyUniverse(BaseModel):
    """Root model for ``config/companies.yaml``."""

    model_config = ConfigDict(extra="forbid")

    defaults: dict[str, Any] = Field(default_factory=dict)
    companies: tuple[CompanyConfig, ...] = ()

    @model_validator(mode="after")
    def _reject_duplicates(self) -> CompanyUniverse:
        from src.utils.normalization import normalize_company_name

        seen: dict[str, str] = {}
        for company in self.companies:
            key = normalize_company_name(company.name)
            if key in seen:
                raise ValueError(
                    f"duplicate company entry: {company.name!r} collides with {seen[key]!r}"
                )
            seen[key] = company.name
        return self

    def enabled_companies(self) -> tuple[CompanyConfig, ...]:
        return tuple(c for c in self.companies if c.enabled)

    def find(self, name: str) -> CompanyConfig | None:
        """Look a company up by display name or alias, case-insensitively."""
        from src.utils.normalization import normalize_company_name

        target = normalize_company_name(name)
        if not target:
            return None
        for company in self.companies:
            if any(normalize_company_name(n) == target for n in company.all_names):
                return company
        return None


# ---------------------------------------------------------------------------
# Secrets (environment only)
# ---------------------------------------------------------------------------


class Secrets(BaseModel):
    """Credentials pulled from the environment.

    Values are never logged; :func:`src.utils.logging.redact` also guards
    against accidental interpolation into a message.
    """

    model_config = ConfigDict(frozen=True)

    llm_provider: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_username: str | None = None
    smtp_password: str | None = None
    notification_email: str | None = None
    notification_from: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Secrets:
        source = env if env is not None else dict(os.environ)

        def get(key: str) -> str | None:
            value = source.get(key)
            if value is None:
                return None
            value = value.strip()
            return value or None

        port_raw = get("SMTP_PORT")
        try:
            port = int(port_raw) if port_raw else None
        except ValueError:
            port = None

        return cls(
            llm_provider=get("LLM_PROVIDER"),
            gemini_api_key=get("GEMINI_API_KEY"),
            gemini_model=get("GEMINI_MODEL"),
            smtp_host=get("SMTP_HOST"),
            smtp_port=port,
            smtp_username=get("SMTP_USERNAME"),
            smtp_password=get("SMTP_PASSWORD"),
            notification_email=get("NOTIFICATION_EMAIL"),
            notification_from=get("NOTIFICATION_FROM") or get("SMTP_USERNAME"),
        )

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_username and self.smtp_password and self.notification_email)

    def missing_smtp_fields(self) -> list[str]:
        required = {
            "SMTP_HOST": self.smtp_host,
            "SMTP_USERNAME": self.smtp_username,
            "SMTP_PASSWORD": self.smtp_password,
            "NOTIFICATION_EMAIL": self.notification_email,
        }
        return [name for name, value in required.items() if not value]


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Everything the pipeline needs to run, assembled and validated."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings
    roles: RolesConfig
    universe: CompanyUniverse
    secrets: Secrets
    config_dir: Path
    project_root: Path
    # CLI switches that alter behaviour without touching the YAML.
    dry_run: bool = False
    fixture_mode: bool = False
    diagnostic: bool = False
    send_email: bool = True
    company_filter: str | None = None

    @property
    def url_policy(self) -> UrlPolicy:
        return self.settings.urls.to_policy()

    @property
    def freshness_hours(self) -> float:
        return self.settings.run.freshness_hours

    def path(self, relative: str) -> Path:
        """Resolve a configured relative path against the project root."""
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    @property
    def fixture_dir(self) -> Path:
        return self.path(self.settings.fixtures.directory)

    def target_companies(self) -> tuple[CompanyConfig, ...]:
        """Companies to crawl, honouring ``--company``."""
        companies = self.universe.enabled_companies()
        if not self.company_filter:
            return companies
        match = self.universe.find(self.company_filter)
        if match is None:
            raise ConfigError(
                f"--company {self.company_filter!r} not found in config/companies.yaml"
            )
        return (match,)

    @property
    def llm_enabled(self) -> bool:
        provider = (self.secrets.llm_provider or self.settings.llm.provider or "none").lower()
        if provider in ("none", "off", "disabled"):
            return False
        if provider == "gemini":
            return bool(self.secrets.gemini_api_key)
        return False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"missing configuration file: {path}")
    try:
        # safe_load never constructs arbitrary Python objects.
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return loaded


def _apply_env_overrides(settings_data: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    """Allow a few high-traffic settings to be overridden by environment vars."""
    run = dict(settings_data.get("run") or {})
    if value := env.get("FRESHNESS_HOURS"):
        try:
            run["freshness_hours"] = float(value)
        except ValueError:
            pass
    if value := env.get("MAX_CONCURRENCY"):
        try:
            run["max_concurrency"] = int(value)
        except ValueError:
            pass
    if run:
        settings_data["run"] = run

    llm = dict(settings_data.get("llm") or {})
    if value := env.get("LLM_PROVIDER"):
        llm["provider"] = value.strip().lower()
    if value := env.get("GEMINI_MODEL"):
        llm["model"] = value.strip()
    if value := env.get("LLM_MAX_CALLS_PER_RUN"):
        try:
            llm["max_calls_per_run"] = int(value)
        except ValueError:
            pass
    if llm:
        settings_data["llm"] = llm
    return settings_data


def _find_project_root(config_dir: Path) -> Path:
    """The directory containing ``config/``, i.e. the repository root."""
    return config_dir.parent if config_dir.name == "config" else config_dir


def load_config(
    config_dir: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
    fixture_mode: bool = False,
    send_email: bool = True,
    company_filter: str | None = None,
    diagnostic: bool = False,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load and validate all configuration.

    ``overrides`` is a shallow-merged dict applied to the parsed ``settings.yaml``
    data, used by the CLI (``--freshness-hours``) and by tests.
    """
    environment = dict(os.environ) if env is None else env
    resolved_dir = Path(
        config_dir or environment.get("JOB_AGENT_CONFIG_DIR") or "config"
    ).expanduser()
    if not resolved_dir.is_absolute():
        resolved_dir = (Path.cwd() / resolved_dir).resolve()

    if not resolved_dir.is_dir():
        raise ConfigError(f"configuration directory not found: {resolved_dir}")

    settings_data = _apply_env_overrides(_read_yaml(resolved_dir / "settings.yaml"), environment)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            merged = dict(settings_data.get(section) or {})
            merged.update(values)
            settings_data[section] = merged
        else:
            settings_data[section] = values

    roles_data = _read_yaml(resolved_dir / "roles.yaml")
    companies_data = _read_yaml(resolved_dir / "companies.yaml")

    try:
        settings = Settings.model_validate(settings_data)
        roles = RolesConfig.model_validate(roles_data)
        universe = CompanyUniverse.model_validate(companies_data)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"configuration validation failed: {exc}") from exc

    return AppConfig(
        settings=settings,
        roles=roles,
        universe=universe,
        secrets=Secrets.from_env(environment),
        config_dir=resolved_dir,
        project_root=_find_project_root(resolved_dir),
        dry_run=dry_run,
        fixture_mode=fixture_mode,
        diagnostic=diagnostic,
        send_email=send_email,
        company_filter=company_filter,
    )
