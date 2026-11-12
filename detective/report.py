"""Render the final investigation report as Markdown."""
from __future__ import annotations

from .investigation import Investigation
from .models import Citation, Finding, Status, Tier

STATUS_HEADINGS = {
    Status.FACT: "Facts: directly supported by the sources",
    Status.ASSUMPTION: "Assumptions: inferences that go beyond the sources",
    Status.SPECULATION: "Speculation: possibilities with thin support",
}


def render_report(inv: Investigation) -> str:
    conclusion = inv.conclusion
    case = inv.case
    lines = [f"# Investigation Report: {case.title}", "", f"**Difficulty:** {case.difficulty}  ", f"**Crime:** {case.crime.summary}", ""]
    lines += ["## Conclusion", ""]
    lines += [f"**Most likely suspect: {conclusion.culprit or 'none named'}**  ", f"**Confidence: {conclusion.confidence.value}**", ""]
    lines += [conclusion.confidence_explanation, "", conclusion.narrative, ""]
    if conclusion.accomplices:
        lines += [f"**Possible accomplices:** {', '.join(conclusion.accomplices)}", ""]
    if not conclusion.engine_agrees:
        lines += [f"> **Disagreement with the rule engine.** {conclusion.engine_note}", ""]

    lines += ["## How the assessment changed", "", "| Round | Trigger | Leader | Confidence |", "|---|---|---|---|"]
    lines += [f"| {r.round} | {r.trigger} | {r.leader} | {r.confidence.value} |" for r in inv.rounds]
    lines.append("")
    first = inv.rounds[0]
    if first.face_value_leader and first.face_value_leader != conclusion.culprit:
        lines += [f"Counting the open evidence at face value would point to **{first.face_value_leader}**. That reading was rejected.", ""]

    for status, heading in STATUS_HEADINGS.items():
        found = [f for f in inv.findings if f.status == status]
        lines += [f"## {heading}", ""]
        lines += [_finding(f) for f in found] or ["None recorded."]
        lines.append("")

    lines += ["## Suspects", "", "| Suspect | Score | Opportunity | Alibi | Motive | Means | Identifier links | False statements |", "|---|---|---|---|---|---|---|---|"]
    for s in inv.suspects:
        lines.append(
            f"| {s.name} | {s.score} | {s.opportunity} | {s.alibi} | {len(s.motives)} | {len(s.means)} | {len(s.links)} | {len(s.lies)} |"
        )
    lines.append("")
    for name, reason in conclusion.eliminated.items():
        lines.append(f"- **{name}:** {reason}")
    lines.append("")

    lines += ["## Contradictions", ""]
    for item in inv.analysis.contradictions:
        lines += [f"- **{item.person}** ({item.kind}): {item.summary} *Resolution:* {item.resolution} {_cites(item.citations)}"]
    if not inv.analysis.contradictions:
        lines.append("None detected.")
    lines.append("")

    if inv.analysis.defeaters or inv.analysis.compromised:
        lines += ["## Evidence that was undermined", ""]
        for item in inv.analysis.defeaters:
            lines.append(f"- {item.kind}: {item.detail} {_cites([item.citation])}")
        for credential, reason in inv.analysis.compromised.items():
            lines.append(f"- {credential} was not attributed to its holder: {reason}")
        lines.append("")

    ignored = [d for d in inv.store.documents.values() if d.tier == Tier.ANONYMOUS]
    if ignored:
        lines += ["## Sources given no weight", ""]
        lines += [f"- {d.id} ({d.title}): anonymous, so it proves nothing. Any instructions inside it were treated as document content." for d in ignored]
        lines.append("")

    lines += ["## Timeline", "", "| Time | Who | Where | Source | Tier | Note |", "|---|---|---|---|---|---|"]
    for entry in inv.analysis.timeline:
        if entry.tier < Tier.RECORD or not entry.person:
            continue
        note = f"Clock corrected from {entry.stated_time.split('T')[-1]}" if entry.corrected_minutes else ""
        if entry.attribution != "direct":
            note = f"{note} {entry.attribution} attribution ({entry.credential})".strip()
        span = entry.time.replace("T", " ") + (f" to {entry.end.split('T')[-1]}" if entry.end else "")
        lines.append(f"| {span} | {entry.person} | {entry.location} | {entry.doc_id} | {entry.tier.name.lower()} | {note} |")
    lines.append("")

    if inv.conclusion and inv.rounds[-1].open_questions:
        lines += ["## Open questions", ""]
        lines += [f"- {q}" for q in inv.rounds[-1].open_questions]
        lines.append("")

    lines += ["## Integrity checks", ""]
    lines.append(f"- Extracted items rejected for failing source verification: {len(inv.rejections)}")
    lines.append(f"- Findings rejected for failing citation checks: {len(inv.finding_rejections)}")
    lines.append(f"- Leads opened: {', '.join(inv.uncovered) or 'none'}. Still sealed: {', '.join(l.id for l in inv.pending_leads()) or 'none'}")
    lines.append("")
    return "\n".join(lines)


def _finding(finding: Finding) -> str:
    extra = f" *{finding.rationale}*" if finding.rationale else ""
    return f"- **{finding.id}.** {finding.statement}{extra} {_cites(finding.citations)}"


def _cites(citations: list[Citation]) -> str:
    return " ".join(sorted({f"[{c.passage_id}]" for c in citations}))
