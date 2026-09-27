"""Synthetic resume PDFs and resume-intelligence tests. No personal resume data."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.agents.resume_intelligence_agent import ResumeFacts, interpret_resume
from src.models.job import RawJobPosting
from src.services.candidate_profile import CandidateProfile, SkillEvidence
from src.services.resume_extract import load_resume_pdf
from src.services.resume_parse import merge_months, parse_spans
from src.services.resume_store import diff_profiles, load_profile, store_profile

ONE_PAGE = """
SUMMARY
Backend engineer who builds APIs.
EXPERIENCE
Software Engineer, Example Labs
Feb 2022 - Aug 2023
- Developed backend services using Java and Spring Boot.
- Built REST APIs with PostgreSQL.
PROJECTS
Campus Portal
- Provisioned AWS infrastructure using Terraform.
EDUCATION
B.S. Computer Science, State University, 2024
SKILLS
Java, Python, Node.js
CERTIFICATIONS
AWS Cloud Practitioner
"""

MULTI_PAGE_A = """
EXPERIENCE
Software Engineer, Example Labs
Feb 2022 - Aug 2023
- Developed backend services using Java and Spring Boot.
"""

MULTI_PAGE_B = """
PROJECTS
Campus Portal
- Provisioned AWS infrastructure using Terraform.
EDUCATION
B.S. Computer Science, State University, 2024
"""

BULLETS_ONLY = """
EXPERIENCE
Software Engineer, Example Labs
Feb 2022 - Aug 2023
- Built services with Spring Boot.
"""

UPDATED = """
EXPERIENCE
Software Engineer, Example Labs
Feb 2022 - Aug 2023
- Developed backend services using Java and Spring Boot.
- Deployed workloads with Kubernetes.
SKILLS
Java, Kubernetes
"""

OVERLAP = """
EXPERIENCE
Software Engineer, Example Labs
Feb 2022 - Aug 2023
- Developed backend services using Java.
Engineer, Other Lab
Jan 2023 - Apr 2023
- Built REST APIs with PostgreSQL.
"""

AMBIGUOUS = """
EXPERIENCE
Intern, Example Labs
Summer 2023
- Built REST APIs with PostgreSQL.
"""


def _pdf(path: Path, pages: list[str]) -> None:
    path.write_bytes(_pdf_bytes(pages))


def _pdf_bytes(pages: list[str]) -> bytes:
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    content_ids: list[int] = []
    for text in pages:
        lines = text.splitlines() or [""]
        commands = ["BT", "/F1 10 Tf", "50 760 Td"]
        for index, line in enumerate(lines[:50]):
            if index:
                commands.append("0 -14 Td")
            safe = line.encode("latin-1", "replace").decode("latin-1")
            safe = safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({safe}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        content_ids.append(add(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"))
    n = len(pages)
    pages_id_expected = n + 2
    kids = " ".join(f"{pages_id_expected + 1 + index} 0 R" for index in range(n))
    pages_id = add(f"<< /Type /Pages /Count {n} /Kids [{kids}] >>".encode())
    assert pages_id == pages_id_expected
    for content_id in content_ids:
        add(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font} 0 R >> >> >>"
            ).encode()
        )
    catalog = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
    return _serialize(objects, catalog)


def _serialize(objects: list[bytes], catalog: int) -> bytes:
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )
    return bytes(output)


def test_pdf_text_and_sections(tmp_path: Path) -> None:
    path = tmp_path / "resume.pdf"
    _pdf(path, [ONE_PAGE])
    document = load_resume_pdf(path)
    assert document.page_count == 1
    assert "Spring Boot" in document.text
    profile = asyncio.run(
        interpret_resume(document.text, filename="resume.pdf", sha256=document.sha256, page_count=1)
    )
    names = {skill.name: skill.status for skill in profile.skills}
    assert names["Java"] == "EXPLICIT"
    assert names["Spring Boot"] == "EXPLICIT"
    assert names["PostgreSQL"] == "EXPLICIT"
    assert "Node.js" in names
    assert names["AWS"] == "EXPLICIT"
    assert names["Terraform"] == "EXPLICIT"
    assert profile.work_experience[0].company == "Example Labs"
    assert profile.education
    assert profile.certifications == ["AWS Cloud Practitioner"]
    assert profile.professional_months == 18
    assert any(item.role_family == "backend_engineering" for item in profile.role_families)


def test_multipage_and_inferred_java(tmp_path: Path) -> None:
    path = tmp_path / "resume.pdf"
    _pdf(path, [MULTI_PAGE_A, MULTI_PAGE_B])
    document = load_resume_pdf(path)
    assert document.page_count == 2
    profile = asyncio.run(interpret_resume(document.text, filename="resume.pdf", sha256="b", page_count=2))
    assert any(skill.name == "AWS" for skill in profile.skills)
    bullet_profile = asyncio.run(interpret_resume(BULLETS_ONLY, filename="resume.pdf", sha256="c", page_count=1))
    java = next(skill for skill in bullet_profile.skills if skill.name == "Java")
    assert java.status == "INFERRED"
    assert "Spring Boot" in java.evidence[0]["quote"]


def test_overlapping_dates_are_not_double_counted() -> None:
    spans = parse_spans(OVERLAP, today=date(2026, 9, 25))
    assert merge_months(spans, kind="professional") == 18


def test_ambiguous_dates_are_not_counted() -> None:
    spans = parse_spans(AMBIGUOUS, today=date(2026, 9, 25))
    assert spans[0].ambiguous or spans[0].start is None
    assert merge_months(spans, kind="internship") == 0
    assert merge_months(spans, kind="professional") == 0


def test_llm_invention_is_dropped() -> None:
    class Scripted:
        available = True

        async def structured(self, **kwargs: object) -> ResumeFacts:
            return ResumeFacts(
                skills=[{"name": "Kafka", "quote": "Expert in Kafka streaming"}],
                education=["Invented University"],
                certifications=["AWS Cloud Practitioner"],
            )

    profile = asyncio.run(
        interpret_resume(ONE_PAGE, filename="resume.pdf", sha256="a", page_count=1, llm=Scripted())
    )
    assert all(skill.name != "Kafka" for skill in profile.skills)
    assert all("Invented" not in line for line in profile.education)
    assert "AWS Cloud Practitioner" in profile.certifications


def test_resume_versions_and_diff(tmp_path: Path) -> None:
    first = asyncio.run(interpret_resume(ONE_PAGE, filename="resume.pdf", sha256="aaa", page_count=1))
    path = tmp_path / "profile.json"
    history = tmp_path / "history"
    assert store_profile(first, path, history) is None
    saved = load_profile(path)
    assert saved is not None and saved.profile_version == 1
    assert store_profile(first, path, history) is None
    assert load_profile(path).profile_version == 1
    second = asyncio.run(interpret_resume(UPDATED, filename="resume.pdf", sha256="bbb", page_count=1))
    change = store_profile(second, path, history)
    assert change is not None
    assert "Kubernetes" in change.added
    updated = load_profile(path)
    assert updated is not None and updated.profile_version == 2
    archived = load_profile(history / "profile_v1.json")
    assert archived is not None and archived.resume.sha256 == "aaa"
    assert (history / "diff_v2.json").exists()


def test_skill_still_present_is_not_removed() -> None:
    old = asyncio.run(interpret_resume(ONE_PAGE, filename="resume.pdf", sha256="aaa", page_count=1))
    new = asyncio.run(interpret_resume(UPDATED, filename="resume.pdf", sha256="bbb", page_count=1))
    change = diff_profiles(old, new)
    assert "Java" not in change.removed
    assert "Kubernetes" in change.added


def test_jobs_match_resume_evidence(tmp_config) -> None:
    profile_doc = asyncio.run(interpret_resume(ONE_PAGE, filename="resume.pdf", sha256="aaa", page_count=1))
    evidence = [
        SkillEvidence(
            skill=skill.name,
            evidence=skill.evidence[0]["quote"],
            source=skill.evidence[0]["source"],
            status=skill.status,
        )
        for skill in profile_doc.skills
    ]
    profile = CandidateProfile(experience_evidence=evidence, max_required_years=2)
    strong = evaluate_job(
        RawJobPosting(
            source="greenhouse",
            company_name="Acme",
            title="Software Engineer",
            description="Build REST APIs in Java. 0-2 years of experience.",
            job_id="1",
            apply_url="https://boards.greenhouse.io/acme/jobs/1",
        ),
        profile,
        tmp_config.roles,
    )
    unrelated = evaluate_job(
        RawJobPosting(
            source="greenhouse",
            company_name="Acme",
            title="Account Manager",
            description="Manage client accounts. 0-2 years.",
            job_id="2",
        ),
        profile,
        tmp_config.roles,
    )
    senior = evaluate_job(
        RawJobPosting(
            source="greenhouse",
            company_name="Acme",
            title="Senior Software Engineer",
            description="5+ years required. Java and PostgreSQL.",
            job_id="3",
        ),
        profile,
        tmp_config.roles,
    )
    missing = evaluate_job(
        RawJobPosting(
            source="greenhouse",
            company_name="Acme",
            title="Software Engineer",
            description="Kafka streaming required. 0-2 years.",
            job_id="4",
        ),
        profile,
        tmp_config.roles,
    )
    assert "Java" in strong.matched_requirements
    critique = review_fit(
        strong,
        RawJobPosting(source="greenhouse", title="Software Engineer", description="Build REST APIs in Java."),
        profile,
    )
    assert "Java" in critique.fit.matched_requirements
    assert unrelated.decision == "REJECT"
    assert senior.decision == "REJECT"
    assert "Kafka" not in missing.matched_requirements
    invented = strong.model_copy(deep=True)
    invented.matched_requirements = ["Java", "Kafka"]
    invented.candidate_evidence.append({"requirement": "Kafka", "evidence": "Invented Kafka experience"})
    cleaned = review_fit(
        invented,
        RawJobPosting(source="greenhouse", title="Software Engineer", description="Build REST APIs in Java."),
        profile,
    )
    assert "Kafka" in cleaned.unsupported_claims
    assert "Kafka" not in cleaned.fit.matched_requirements


def test_resume_profile_is_what_matching_uses(tmp_path: Path, tmp_config) -> None:
    from src.services.candidate_profile import load_candidate_profile, merge_resume_profile

    profile_doc = asyncio.run(interpret_resume(ONE_PAGE, filename="resume.pdf", sha256="aaa", page_count=1))
    path = tmp_path / "profile.json"
    store_profile(profile_doc, path, tmp_path / "history")
    search = load_candidate_profile(tmp_config.project_root / "config" / "candidate_profile.yaml")
    search = search.model_copy(update={"technical_skills": ["Should Not Be Used"]})
    merged = merge_resume_profile(search, path)
    assert merged.technical_skills == []
    assert merged.profile_version == 1
    assert merged.professional_months == 18
    assert any(item.skill == "Java" and "Java" in item.evidence for item in merged.experience_evidence)
    fit = evaluate_job(
        RawJobPosting(
            source="greenhouse",
            company_name="Acme",
            title="Software Engineer",
            description="Build REST APIs in Java. 0-2 years of experience.",
            job_id="1",
        ),
        merged,
        tmp_config.roles,
    )
    java = next(item for item in fit.candidate_evidence if item["requirement"] == "Java")
    assert java["profile_version"] == "1"
    assert "Should Not Be Used" not in fit.matched_requirements


def test_critic_rejects_years_beyond_resume_employment(tmp_config) -> None:
    profile = CandidateProfile(
        professional_months=6,
        experience_evidence=[
            SkillEvidence(
                skill="Kubernetes",
                evidence="Used Kubernetes in a class project.",
                source="projects",
                status="EXPLICIT",
            )
        ],
    )
    posting = RawJobPosting(
        source="greenhouse",
        company_name="Acme",
        title="Software Engineer",
        description="5+ years of Kubernetes experience required.",
        job_id="9",
    )
    rejected = evaluate_job(posting, profile, tmp_config.roles)
    assert rejected.decision == "REJECT"
    claimed = rejected.model_copy(update={"decision": "STRONG_MATCH", "experience_alignment": "strong"})
    critique = review_fit(claimed, posting, profile)
    assert critique.fit.experience_alignment == "reject"
    assert "required years exceed resume employment" in critique.issues
    assert critique.approved is False
    invented = claimed.model_copy(deep=True)
    invented.candidate_evidence = [{"requirement": "Kubernetes", "evidence": "5 years from a project"}]
    cleaned = review_fit(invented, posting, profile)
    assert "Kubernetes" in cleaned.unsupported_claims


def test_metrics_and_contact_stay_explicit() -> None:
    text = """
Name: Jordan Example
jordan.example@example.com
(617) 555-0100
EXPERIENCE
Software Engineer, Example Labs
Feb 2022 - Aug 2023
- reduced processing time by 80%
- Developed backend services using Java.
- Developed backend services using Java.
"""
    profile = asyncio.run(interpret_resume(text, filename="resume.pdf", sha256="m", page_count=1))
    assert profile.contact_name == "Jordan Example"
    assert profile.contact_email == "jordan.example@example.com"
    assert profile.contact_phone == "(617) 555-0100"
    assert profile.work_experience[0].achievements == ["reduced processing time by 80%"]
    java = next(skill for skill in profile.skills if skill.name == "Java")
    assert len(java.evidence) == 1
    assert java.evidence[0]["company"] == "Example Labs"


