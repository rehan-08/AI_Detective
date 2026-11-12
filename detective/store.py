"""Searchable evidence store.

BM25 over passages rather than embeddings: a case file is small, names and
identifiers matter more than paraphrase, and lexical scores are deterministic and
explainable, which matters when a retrieval result has to be defended as evidence.
Light stemming and alias expansion cover the most common vocabulary gaps.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .ingest import split_passages
from .models import Case, Document, Passage

_STOP = set(
    "a an and are as at be but by did do for from had has have he her his i in is it its me my no not of on or "
    "our she so that the their them they this to was we were what when where which who why will with would you".split()
)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9']*")
# Log vocabulary rarely matches how investigators phrase a question.
_SYNONYMS = {"entry": "enter", "exit": "leave", "left": "leave", "departed": "leave", "re": ""}


def _stem(token: str) -> str:
    for suffix in ("ings", "ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    tokens = (_SYNONYMS.get(t, t) for t in _TOKEN.findall(text.lower()) if t not in _STOP)
    return [_stem(t) for t in tokens if t]


class EvidenceStore:
    K1, B = 1.5, 0.75

    def __init__(self, case: Case):
        self.case = case
        self.documents: dict[str, Document] = {}
        self.passages: dict[str, Passage] = {}
        self._tf: dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._length: dict[str, int] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._aliases = _alias_map(case)
        for doc in case.documents:
            self.add_document(doc)

    def add_document(self, doc: Document) -> list[Passage]:
        if doc.id in self.documents:
            return []
        self.documents[doc.id] = doc
        passages = split_passages(doc)
        title_tokens = tokenize(doc.title)
        for passage in passages:
            self.passages[passage.id] = passage
            terms = Counter(tokenize(passage.text) + title_tokens)
            self._tf[passage.id] = terms
            self._length[passage.id] = sum(terms.values())
            for term in terms:
                self._df[term] += 1
                self._postings[term].add(passage.id)
        return passages

    def doc_passages(self, doc_id: str) -> list[Passage]:
        return [p for p in self.passages.values() if p.doc_id == doc_id]

    def search(self, query: str, k: int = 5) -> list[tuple[Passage, float]]:
        terms = self._expand(tokenize(query))
        if not self._length:
            return []
        n = len(self._length)
        avg = sum(self._length.values()) / n
        scores: dict[str, float] = defaultdict(float)
        for term in terms:
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for pid in self._postings[term]:
                tf = self._tf[pid][term]
                norm = tf + self.K1 * (1 - self.B + self.B * self._length[pid] / avg)
                scores[pid] += idf * tf * (self.K1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self.passages[pid], round(score, 3)) for pid, score in ranked[:k]]

    def passages_mentioning(self, name: str) -> list[Passage]:
        terms = [name.lower()] + self._aliases.get(name.lower(), [])
        return [p for p in self.passages.values() if any(term in p.text.lower() for term in terms)]

    def _expand(self, terms: list[str]) -> list[str]:
        expanded = list(terms)
        for alias, group in self._aliases.items():
            alias_tokens = tokenize(alias)
            if alias_tokens and all(t in terms for t in alias_tokens):
                for other in group:
                    expanded.extend(tokenize(other))
        return expanded


def _alias_map(case: Case) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for entity in [*case.everyone(), *case.locations]:
        group = [entity.name.lower()] + [a.lower() for a in entity.aliases]
        for name in group:
            aliases[name] = [other for other in group if other != name]
    return aliases
