"""Discovery source adapters.

Each adapter is isolated. A site-structure or API change is fixed in one file
and cannot abort the rest of the run.
"""

from src.sources.ashby import AshbySource
from src.sources.base import DiscoverySource, HttpClient, SourceContext, SourceError, SourceResult
from src.sources.company_career import CompanyCareerSource
from src.sources.greenhouse import GreenhouseSource
from src.sources.icims import IcimsSource
from src.sources.jobright import JobrightSource
from src.sources.lever import LeverSource
from src.sources.smartrecruiters import SmartRecruitersSource
from src.sources.workday import WorkdaySource

__all__ = [
    "COMPANY_SOURCES",
    "GLOBAL_SOURCES",
    "DiscoverySource",
    "HttpClient",
    "SourceContext",
    "SourceError",
    "SourceResult",
    "all_source_classes",
]

GLOBAL_SOURCES: tuple[type[DiscoverySource], ...] = (JobrightSource,)

COMPANY_SOURCES: tuple[type[DiscoverySource], ...] = (
    GreenhouseSource,
    LeverSource,
    AshbySource,
    SmartRecruitersSource,
    WorkdaySource,
    IcimsSource,
    CompanyCareerSource,
)


def all_source_classes() -> tuple[type[DiscoverySource], ...]:
    return (*GLOBAL_SOURCES, *COMPANY_SOURCES)
