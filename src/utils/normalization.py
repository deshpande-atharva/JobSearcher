"""Deterministic text normalization.

Everything here is pure, synchronous and cheap. The point is to resolve the
large majority of postings without an LLM call, so the model is reserved for
genuinely ambiguous text.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

from src.models.job import EmploymentType, RemoteType

__all__ = [
    "ExperienceRequirement",
    "LocationInfo",
    "US_STATE_ABBREVIATIONS",
    "clean_text",
    "detect_employment_type",
    "detect_remote_type",
    "extract_experience_requirement",
    "html_to_text",
    "normalize_company_name",
    "normalize_location",
    "normalize_location_key",
    "normalize_title",
    "split_required_and_preferred",
    "truncate",
]


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_BREAK_RE = re.compile(r"</(p|div|li|tr|h[1-6]|ul|ol|section)\s*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\u00a0]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def html_to_text(raw: str | None) -> str:
    """Flatten an HTML fragment into readable plain text.

    Scraped markup is treated strictly as data: tags are stripped, nothing is
    executed, and no external references are followed.
    """
    if not raw:
        return ""
    text = _SCRIPT_STYLE_RE.sub(" ", raw)
    text = _BR_RE.sub("\n", text)
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NEWLINES_RE.sub("\n\n", text).strip()


def clean_text(raw: str | None) -> str:
    """Collapse whitespace and normalize unicode punctuation."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return _WS_RE.sub(" ", text).strip()


def truncate(text: str | None, limit: int) -> str:
    """Trim to ``limit`` characters with an ellipsis marker."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "\u2026"


# ---------------------------------------------------------------------------
# Company / title normalization
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = (
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",
    "limited",
    "ltd",
    "llc",
    "llp",
    "lp",
    "plc",
    "gmbh",
    "sa",
    "nv",
    "ag",
    "holdings",
    "group",
    "technologies",
    "technology",
)

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_company_name(name: str | None) -> str:
    """Canonical lowercase key for a company.

    Strips punctuation and trailing legal suffixes so ``Visa Inc.``,
    ``Visa, Inc`` and ``VISA`` all collapse to ``visa``. Used for deduplication
    and for aligning postings with historical H-1B employer names.
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text).lower()
    tokens = [t for t in text.split() if t]
    while len(tokens) > 1 and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


_TITLE_NOISE_RE = re.compile(
    r"""
    \(
      [^)]*
      (?:remote|hybrid|onsite|on-site|contract|full[\s-]?time|part[\s-]?time
        |\d{4}|us|usa|united\s+states|multiple\s+locations|new\s+grad)
      [^)]*
    \)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_REQ_ID_RE = re.compile(r"\b(?:req|requisition|job)\s*(?:id|#|no\.?)?\s*[:\-]?\s*[A-Z0-9\-]{4,}\b", re.IGNORECASE)


def normalize_title(title: str | None) -> str:
    """Canonical lowercase form of a job title, used for matching and dedup."""
    if not title:
        return ""
    text = clean_text(title).lower()
    text = _TITLE_NOISE_RE.sub(" ", text)
    text = _REQ_ID_RE.sub(" ", text)
    # Drop trailing location/segment qualifiers separated by dashes or pipes.
    text = re.sub(r"\s*[|/]\s*", " ", text)
    text = _PUNCT_RE.sub(" ", text.replace("+", " plus "))
    text = re.sub(r"\bsr\b", "senior", text)
    text = re.sub(r"\bjr\b", "junior", text)
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Location normalization
# ---------------------------------------------------------------------------

US_STATE_ABBREVIATIONS: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "VI": "Virgin Islands",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

_STATE_NAME_TO_ABBREV: dict[str, str] = {
    name.lower(): abbrev for abbrev, name in US_STATE_ABBREVIATIONS.items()
}

_US_COUNTRY_TOKENS = (
    "united states",
    "united states of america",
    "usa",
    "u.s.a.",
    "u.s.",
    " us",
    "us ",
    "america",
)

# Countries and cities that appear often in global postings. Presence of one of
# these with no U.S. signal marks the posting international-only.
_NON_US_TOKENS = (
    "canada",
    "toronto",
    "vancouver",
    "montreal",
    "ontario",
    "british columbia",
    "quebec",
    "mexico",
    "guadalajara",
    "brazil",
    "sao paulo",
    "argentina",
    "united kingdom",
    "england",
    "london",
    "manchester",
    "scotland",
    "ireland",
    "dublin",
    "germany",
    "berlin",
    "munich",
    "france",
    "paris",
    "spain",
    "madrid",
    "barcelona",
    "portugal",
    "lisbon",
    "netherlands",
    "amsterdam",
    "belgium",
    "brussels",
    "switzerland",
    "zurich",
    "sweden",
    "stockholm",
    "norway",
    "oslo",
    "denmark",
    "copenhagen",
    "finland",
    "helsinki",
    "poland",
    "warsaw",
    "krakow",
    "czech",
    "prague",
    "romania",
    "bucharest",
    "italy",
    "milan",
    "rome",
    "austria",
    "vienna",
    "israel",
    "tel aviv",
    "india",
    "bangalore",
    "bengaluru",
    "hyderabad",
    "pune",
    "chennai",
    "gurgaon",
    "gurugram",
    "noida",
    "mumbai",
    "delhi",
    "china",
    "beijing",
    "shanghai",
    "shenzhen",
    "hong kong",
    "taiwan",
    "taipei",
    "japan",
    "tokyo",
    "osaka",
    "korea",
    "seoul",
    "singapore",
    "malaysia",
    "kuala lumpur",
    "philippines",
    "manila",
    "indonesia",
    "jakarta",
    "vietnam",
    "hanoi",
    "thailand",
    "bangkok",
    "australia",
    "sydney",
    "melbourne",
    "new zealand",
    "auckland",
    "south africa",
    "cape town",
    "johannesburg",
    "nigeria",
    "lagos",
    "kenya",
    "nairobi",
    "egypt",
    "cairo",
    "uae",
    "dubai",
    "abu dhabi",
    "saudi arabia",
    "riyadh",
    "turkey",
    "istanbul",
    "emea",
    "apac",
    "latam",
)

_REMOTE_RE = re.compile(
    r"\b(?:fully\s+remote|remote|work\s+from\s+home|wfh|telecommute|virtual|distributed)\b",
    re.IGNORECASE,
)
_HYBRID_RE = re.compile(r"\b(?:hybrid|flexible\s+work|partially\s+remote|\d\s*days?\s+(?:in|per)\s+office)\b", re.IGNORECASE)
_ONSITE_RE = re.compile(r"\b(?:on-?site|in-?office|in-?person)\b", re.IGNORECASE)

_CITY_STATE_RE = re.compile(
    r"([A-Za-z][A-Za-z.\-' ]+?)\s*,\s*([A-Za-z]{2}|[A-Za-z][A-Za-z ]+?)\b(?:\s*,\s*(?:USA?|United States(?: of America)?))?\s*$",
)


@dataclass(slots=True)
class LocationInfo:
    """Structured view of a free-text location string."""

    raw: str
    display: str
    city: str | None = None
    state: str | None = None
    is_us: bool = False
    is_international_only: bool = False
    remote_type: RemoteType = RemoteType.UNKNOWN
    is_multiple: bool = False


def detect_remote_type(*texts: str | None) -> RemoteType:
    """Classify the work arrangement from any combination of text fields.

    Hybrid is checked first: "Hybrid - 3 days remote" mentions remote but is not
    a remote job.
    """
    blob = " ".join(t for t in texts if t)
    if not blob.strip():
        return RemoteType.UNKNOWN
    if _HYBRID_RE.search(blob):
        return RemoteType.HYBRID
    if _REMOTE_RE.search(blob):
        return RemoteType.REMOTE
    if _ONSITE_RE.search(blob):
        return RemoteType.ONSITE
    return RemoteType.UNKNOWN


def normalize_location(raw: str | None, *, description: str | None = None) -> LocationInfo:
    """Parse a location string into city/state plus U.S. and remote signals.

    ``is_us`` is only set when there is positive evidence (a U.S. state, a U.S.
    country token, or explicitly U.S.-scoped remote wording).
    ``is_international_only`` is set when non-U.S. signals appear with no U.S.
    signal at all, which is the condition the location filter rejects on.
    """
    text = clean_text(raw)
    if not text:
        return LocationInfo(
            raw="",
            display="Unknown",
            remote_type=detect_remote_type(description),
        )

    lowered = text.lower()
    remote_type = detect_remote_type(text) 
    if remote_type is RemoteType.UNKNOWN:
        remote_type = detect_remote_type(description)

    is_multiple = bool(
        re.search(r"\b(?:multiple|various)\s+(?:locations|cities)\b", lowered)
        or lowered.count(";") >= 1
        or lowered.count(" or ") >= 1
    )

    city: str | None = None
    state: str | None = None

    # Strip remote/hybrid prefixes so "Remote - Austin, TX" still parses.
    candidate = re.sub(
        r"^\s*(?:remote|hybrid|on-?site)\s*[-\u2013:|,]\s*", "", text, flags=re.IGNORECASE
    )
    # Use the first segment when several locations are listed.
    first_segment = re.split(r"\s*(?:;|\bor\b|\||/)\s*", candidate)[0].strip()

    match = _CITY_STATE_RE.search(first_segment)
    if match:
        raw_city = match.group(1).strip()
        raw_state = match.group(2).strip()
        upper_state = raw_state.upper()
        if upper_state in US_STATE_ABBREVIATIONS:
            city, state = raw_city, upper_state
        elif raw_state.lower() in _STATE_NAME_TO_ABBREV:
            city, state = raw_city, _STATE_NAME_TO_ABBREV[raw_state.lower()]

    if state is None:
        # A bare state name anywhere in the string, e.g. "Texas, United States".
        for name, abbrev in _STATE_NAME_TO_ABBREV.items():
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                state = abbrev
                break

    has_us_country_token = any(token in lowered for token in _US_COUNTRY_TOKENS) or bool(
        re.search(r"\b(?:us|usa|u\.s\.a?\.?)\b", lowered)
    )
    has_non_us_token = any(
        re.search(rf"\b{re.escape(token)}\b", lowered) for token in _NON_US_TOKENS
    )

    is_us = bool(state) or has_us_country_token
    # "Remote" with a U.S. country token is a U.S. remote role. Bare "Remote"
    # stays unknown-country and is resolved later by the location agent.
    is_international_only = has_non_us_token and not is_us

    return LocationInfo(
        raw=text,
        display=text,
        city=city,
        state=state,
        is_us=is_us,
        is_international_only=is_international_only,
        remote_type=remote_type,
        is_multiple=is_multiple,
    )


def normalize_location_key(raw: str | None) -> str:
    """Location key used in deduplication fingerprints."""
    info = normalize_location(raw)
    if info.city and info.state:
        return f"{info.city.lower()}|{info.state.lower()}"
    return normalize_company_name(raw) or (info.display or "").lower()


# ---------------------------------------------------------------------------
# Employment type
# ---------------------------------------------------------------------------

_EMPLOYMENT_PATTERNS: tuple[tuple[EmploymentType, re.Pattern[str]], ...] = (
    (EmploymentType.CO_OP, re.compile(r"\b(?:co-?op|cooperative\s+education)\b", re.IGNORECASE)),
    (EmploymentType.INTERNSHIP, re.compile(r"\b(?:intern|internship|summer\s+analyst)\b", re.IGNORECASE)),
    (
        EmploymentType.CONTRACT,
        re.compile(r"\b(?:contract|contractor|contract-to-hire|c2h|1099|w2\s+contract|consultant)\b", re.IGNORECASE),
    ),
    (EmploymentType.TEMPORARY, re.compile(r"\b(?:temporary|temp|seasonal)\b", re.IGNORECASE)),
    (EmploymentType.PART_TIME, re.compile(r"\bpart[\s-]?time\b", re.IGNORECASE)),
    (EmploymentType.VOLUNTEER, re.compile(r"\bvolunteer\b", re.IGNORECASE)),
    (EmploymentType.FULL_TIME, re.compile(r"\b(?:full[\s-]?time|permanent|regular|fte)\b", re.IGNORECASE)),
)

# Schema.org / ATS codes seen in structured payloads.
_EMPLOYMENT_CODES: dict[str, EmploymentType] = {
    "full_time": EmploymentType.FULL_TIME,
    "fulltime": EmploymentType.FULL_TIME,
    "ft": EmploymentType.FULL_TIME,
    "regular": EmploymentType.FULL_TIME,
    "permanent": EmploymentType.FULL_TIME,
    "part_time": EmploymentType.PART_TIME,
    "parttime": EmploymentType.PART_TIME,
    "pt": EmploymentType.PART_TIME,
    "contractor": EmploymentType.CONTRACT,
    "contract": EmploymentType.CONTRACT,
    "temporary": EmploymentType.TEMPORARY,
    "temp": EmploymentType.TEMPORARY,
    "intern": EmploymentType.INTERNSHIP,
    "internship": EmploymentType.INTERNSHIP,
    "volunteer": EmploymentType.VOLUNTEER,
    "co_op": EmploymentType.CO_OP,
    "coop": EmploymentType.CO_OP,
    "other": EmploymentType.UNKNOWN,
}


def detect_employment_type(
    explicit: str | None = None,
    *,
    title: str | None = None,
    description: str | None = None,
) -> EmploymentType:
    """Determine employment type, preferring the source's explicit value.

    Title is consulted before description because "Software Engineer Intern"
    is far more reliable than a stray "internship" mention in benefits copy.
    """
    if explicit:
        key = re.sub(r"[^a-z_]", "", explicit.strip().lower().replace("-", "_").replace(" ", "_"))
        if key in _EMPLOYMENT_CODES:
            return _EMPLOYMENT_CODES[key]
        for employment_type, pattern in _EMPLOYMENT_PATTERNS:
            if pattern.search(explicit):
                return employment_type

    if title:
        for employment_type, pattern in _EMPLOYMENT_PATTERNS:
            if pattern.search(title):
                return employment_type

    if description:
        # Only trust an early, explicit statement in the description.
        head = description[:1500]
        for employment_type, pattern in _EMPLOYMENT_PATTERNS:
            if pattern.search(head):
                return employment_type

    return EmploymentType.UNKNOWN


# ---------------------------------------------------------------------------
# Experience requirements
# ---------------------------------------------------------------------------

_PREFERRED_HEADING_RE = re.compile(
    r"""
    ^\s*
    (?:
        preferred(?:\s+(?:qualifications|skills|experience|requirements))?
      | nice[\s-]to[\s-]have s?
      | bonus(?:\s+points)?
      | desired(?:\s+qualifications|\s+skills)?
      | pluses?
      | additional(?:\s+preferred)?\s+qualifications
      | what\s+(?:would\s+)?set s?\s+you\s+apart
      | ideally
    )
    \s*:?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_REQUIRED_HEADING_RE = re.compile(
    r"""
    ^\s*
    (?:
        (?:minimum|basic|required|core)\s+(?:qualifications|requirements|skills|experience)
      | requirements
      | qualifications
      | what\s+you(?:'ll|\s+will)?\s+need
      | who\s+you\s+are
      | must\s+haves?
      | responsibilities
      | about\s+the\s+role
    )
    \s*:?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PREFERRED_INLINE_RE = re.compile(
    r"\b(?:preferred|a\s+plus|nice\s+to\s+have|bonus|ideally|desirable|advantageous)\b",
    re.IGNORECASE,
)

# "0-2 years", "2+ years", "at least 3 years", "minimum 5 years", "two years"
_RANGE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*(?:\+|-|\u2013|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_PLUS_RE = re.compile(
    r"(?:(?:at\s+least|minimum(?:\s+of)?|min\.?|over|more\s+than|>\s*=?)\s*)?(?<!\d)(\d{1,2})\s*\+\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_MIN_RE = re.compile(
    r"(?:at\s+least|minimum(?:\s+of)?|min\.?|no\s+less\s+than|over|more\s+than)\s*(\d{1,2})\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_UPTO_RE = re.compile(
    r"(?:up\s+to|less\s+than|under|fewer\s+than|no\s+more\s+than|<\s*=?)\s*(\d{1,2})\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_BARE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+|professional\s+|industry\s+|work\s+)?experience", re.IGNORECASE)

_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_WORD_YEARS_RE = re.compile(
    rf"\b({'|'.join(_WORD_NUMBERS)})\s*(?:\(\d+\)\s*)?(?:\+\s*)?(?:years?|yrs?)\b", re.IGNORECASE
)


@dataclass(slots=True)
class ExperienceRequirement:
    """Years of experience parsed out of a posting.

    ``min_years`` / ``max_years`` describe the *required* band only. Anything
    found under a "Preferred"/"Nice to have" heading lands in
    ``preferred_min_years`` and must not reject a job on its own.
    """

    min_years: float | None = None
    max_years: float | None = None
    preferred_min_years: float | None = None
    entry_level_signals: list[str] = field(default_factory=list)
    required_quotes: list[str] = field(default_factory=list)
    preferred_quotes: list[str] = field(default_factory=list)
    ambiguous: bool = False

    @property
    def has_required_signal(self) -> bool:
        return self.min_years is not None or self.max_years is not None


def split_required_and_preferred(description: str | None) -> tuple[str, str]:
    """Split a description into required and preferred/optional sections.

    Heading-driven, with an inline fallback: a bullet containing "preferred" or
    "a plus" is treated as preferred even under a required heading.
    """
    if not description:
        return "", ""

    required_lines: list[str] = []
    preferred_lines: list[str] = []
    in_preferred = False

    for line in description.split("\n"):
        stripped = line.strip().lstrip("-*\u2022\u25cf ").strip()
        if not stripped:
            continue
        # Headings are short lines; avoid mis-reading a long sentence as one.
        if len(stripped) <= 80:
            if _PREFERRED_HEADING_RE.match(stripped):
                in_preferred = True
                continue
            if _REQUIRED_HEADING_RE.match(stripped):
                in_preferred = False
                continue
        if in_preferred or _PREFERRED_INLINE_RE.search(stripped):
            preferred_lines.append(stripped)
        else:
            required_lines.append(stripped)

    return "\n".join(required_lines), "\n".join(preferred_lines)


def _scan_years(text: str) -> tuple[float | None, float | None, list[str]]:
    """Extract the tightest (min, max) years signal from a block of text."""
    if not text:
        return None, None, []

    quotes: list[str] = []
    minimums: list[float] = []
    maximums: list[float] = []

    for match in _RANGE_RE.finditer(text):
        low, high = float(match.group(1)), float(match.group(2))
        minimums.append(min(low, high))
        maximums.append(max(low, high))
        quotes.append(match.group(0).strip())

    for match in _PLUS_RE.finditer(text):
        minimums.append(float(match.group(1)))
        quotes.append(match.group(0).strip())

    for match in _MIN_RE.finditer(text):
        minimums.append(float(match.group(1)))
        quotes.append(match.group(0).strip())

    for match in _UPTO_RE.finditer(text):
        maximums.append(float(match.group(1)))
        quotes.append(match.group(0).strip())

    if not minimums and not maximums:
        for match in _BARE_RE.finditer(text):
            minimums.append(float(match.group(1)))
            quotes.append(match.group(0).strip())
        for match in _WORD_YEARS_RE.finditer(text):
            value = _WORD_NUMBERS[match.group(1).lower()]
            minimums.append(float(value))
            quotes.append(match.group(0).strip())

    # The lowest stated minimum is the real bar. Postings often list several
    # alternative paths ("2 years with a BS, or 0 years with an MS").
    min_years = min(minimums) if minimums else None
    max_years = max(maximums) if maximums else None
    return min_years, max_years, quotes[:5]


def extract_experience_requirement(
    description: str | None,
    *,
    title: str | None = None,
    entry_level_signals: tuple[str, ...] | list[str] = (),
) -> ExperienceRequirement:
    """Parse required vs preferred experience from a posting.

    Returns ``ambiguous=True`` when no years and no entry-level wording were
    found, which is the signal the seniority agent uses to escalate to the LLM.
    """
    required_text, preferred_text = split_required_and_preferred(description)
    blob = f"{title or ''}\n{description or ''}".lower()

    found_signals = [signal for signal in entry_level_signals if signal and signal.lower() in blob]

    min_years, max_years, required_quotes = _scan_years(required_text)
    preferred_min, _, preferred_quotes = _scan_years(preferred_text)

    # Title-level numbers ("Software Engineer, 2+ years") count as required.
    if title:
        title_min, title_max, title_quotes = _scan_years(title)
        if title_min is not None and (min_years is None or title_min < min_years):
            min_years = title_min
        if title_max is not None and (max_years is None or title_max > max_years):
            max_years = title_max
        required_quotes.extend(title_quotes)

    ambiguous = min_years is None and max_years is None and not found_signals

    return ExperienceRequirement(
        min_years=min_years,
        max_years=max_years,
        preferred_min_years=preferred_min,
        entry_level_signals=found_signals,
        required_quotes=required_quotes[:5],
        preferred_quotes=preferred_quotes[:5],
        ambiguous=ambiguous,
    )
