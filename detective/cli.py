"""Command-line interface.

  python -m detective list
  python -m detective solve case_02_aldermere_necklace --backend claude
  python -m detective ask case_02_aldermere_necklace "Who signed out FOB-3?"
  python -m detective uncover case_02_aldermere_necklace L1
  python -m detective evaluate --backend offline
  python -m detective serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation import format_table, load_solution, score
from .ingest import load_case
from .investigation import Investigation
from .report import render_report
from .workspace import Backends, Conflict

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"
RUNS_DIR = Path("runs")


def build(case_dir: Path, backend: str) -> tuple[Investigation, object]:
    case = load_case(case_dir)
    try:
        extractor, reasoner = Backends().build(backend, case)
    except Conflict as error:
        sys.exit(str(error))
    return Investigation(case, extractor), reasoner


def case_path(name: str) -> Path:
    path = CASES_DIR / name
    if not (path / "case.json").exists():
        sys.exit(f"Unknown case {name}. Run 'python -m detective list'.")
    return path


def state_path(name: str) -> Path:
    return RUNS_DIR / name / "state.json"


def load_uncovered(name: str) -> list[str]:
    path = state_path(name)
    return json.loads(path.read_text(encoding="utf-8"))["uncovered"] if path.exists() else []


def save_run(name: str, inv: Investigation) -> Path:
    folder = RUNS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "state.json").write_text(json.dumps({"uncovered": inv.uncovered}), encoding="utf-8")
    (folder / "report.md").write_text(render_report(inv), encoding="utf-8")
    trace = [step.model_dump() for step in inv.steps]
    (folder / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return folder


def restore(name: str, backend: str) -> tuple[Investigation, object]:
    inv, reasoner = build(case_path(name), backend)
    for lead_id in load_uncovered(name):
        inv.uncover(lead_id, "Restored from the saved session.")
    return inv, reasoner


def cmd_list(_args) -> None:
    for path in sorted(CASES_DIR.iterdir()):
        if (path / "case.json").exists():
            case = load_case(path)
            print(f"{case.id}  [{case.difficulty}]  {case.title}: {len(case.documents)} documents, {len(case.leads)} sealed leads")


def cmd_solve(args) -> None:
    inv, reasoner = restore(args.case, args.backend)
    conclusion = reasoner.investigate(inv, auto_leads=not args.no_leads)
    folder = save_run(args.case, inv)
    print(f"Most likely suspect: {conclusion.culprit}  Confidence: {conclusion.confidence.value}")
    print(conclusion.confidence_explanation)
    print(f"Full report: {folder / 'report.md'}")


def cmd_uncover(args) -> None:
    inv, reasoner = restore(args.case, args.backend)
    before = inv.begin()
    try:
        after = inv.uncover(args.lead, "Requested by the investigator.")
    except KeyError:
        sys.exit(f"{args.lead} is not a pending lead. Pending: {', '.join(l.id for l in inv.pending_leads()) or 'none'}")
    conclusion = reasoner.investigate(inv, auto_leads=False)
    save_run(args.case, inv)
    print(f"Before: {before.leader} ({before.confidence.value})")
    print(f"After:  {conclusion.culprit} ({conclusion.confidence.value})")
    print(after.confidence_explanation)


def cmd_ask(args) -> None:
    inv, reasoner = restore(args.case, args.backend)
    answer = reasoner.answer(inv, args.question)
    print(f"[{answer.status.value}] {answer.answer}")
    for citation in answer.citations:
        print(f"  {citation.passage_id}: {citation.quote}")


def cmd_evaluate(args) -> None:
    scores = []
    for path in sorted(CASES_DIR.iterdir()):
        if (path / "case.json").exists() and (not args.case or args.case == path.name):
            inv, reasoner = build(path, args.backend)
            reasoner.investigate(inv)
            scores.append(score(inv, load_solution(path)))
    print(format_table(scores))
    for result in scores:
        for miss in result.misses:
            print(f"{result.case}: missed {miss}")
    if args.output:
        Path(args.output).write_text(json.dumps([s.as_dict() for s in scores], indent=2), encoding="utf-8")
    sys.exit(0 if all(s.passed for s in scores) else 1)


def cmd_serve(args) -> None:
    from .server import serve

    print(f"AI Detective is running at http://{args.host}:{args.port}")
    serve(args.host, args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="detective", description="Evidence-grounded AI detective")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list").set_defaults(run=cmd_list)
    serve = commands.add_parser("serve", help="Start the web application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(run=cmd_serve)

    def add(name, handler):
        sub = commands.add_parser(name)
        sub.add_argument("--backend", choices=["offline", "claude", "groq"], default="offline")
        sub.set_defaults(run=handler)
        return sub

    solve = add("solve", cmd_solve)
    solve.add_argument("case")
    solve.add_argument("--no-leads", action="store_true", help="Stay within the open file and report what it supports")
    ask = add("ask", cmd_ask)
    ask.add_argument("case")
    ask.add_argument("question")
    uncover = add("uncover", cmd_uncover)
    uncover.add_argument("case")
    uncover.add_argument("lead")
    evaluate = add("evaluate", cmd_evaluate)
    evaluate.add_argument("--case")
    evaluate.add_argument("--output")
    args = parser.parse_args()
    args.run(args)
