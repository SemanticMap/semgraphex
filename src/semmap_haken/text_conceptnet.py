"""Offset-preserving word mapping to ConceptNet Local (English, optional extra).

The upstream text extractor skips stopwords and probes phrases. We intentionally
use its public existence/edge APIs instead, with our own unchanged-text offsets.
No network access, downloads, database writes or embedding lookups occur here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import sqlite3
import unicodedata


@dataclass(frozen=True)
class Word:
    start: int
    end: int
    surface: str
    lemma: str | None = None


def words(text: str, nlp=None) -> list[Word]:
    """Code-point offsets, [start,end); hyphens split, internal apostrophes stay.

    An explicitly selected spaCy model supplies its own tokenization and lemmas.
    Without it, every Unicode letter/number word is kept, including stopwords.
    """
    if nlp is not None:
        return [Word(t.idx, t.idx + len(t.text), t.text, t.lemma_ or None)
                for t in nlp(text) if any(c.isalnum() for c in t.text)]
    result = []
    i = 0
    while i < len(text):
        if not text[i].isalnum():
            i += 1
            continue
        start = i
        i += 1
        while i < len(text):
            if text[i].isalnum() or unicodedata.category(text[i]).startswith('M'):
                i += 1
            elif text[i] in "'’" and i + 1 < len(text) and text[i + 1].isalnum():
                i += 1
            else:
                break
        result.append(Word(start, i, text[start:i]))
    return result


def concept_uri(term: str) -> str:
    """Upstream English URI convention; never alter the original text."""
    return '/c/en/' + unicodedata.normalize('NFC', term).lower().replace(' ', '_')


class ConceptNetLocal:
    """One read-only connection per worker; bounded per-document existence cache."""
    def __init__(self, database: str | Path, language: str = 'en'):
        if language != 'en':
            raise ValueError('ConceptNet Local supports English only; language must be en')
        try:
            import conceptnet_local as cnl
        except LookupError as exc:
            raise RuntimeError('ConceptNet Local needs NLTK stopwords. Run '
                               '`python -m nltk.downloader stopwords` during setup.') from exc
        except ImportError as exc:
            raise RuntimeError('Install the text extra with Python >=3.12: '
                               "pip install -e '.[text]'") from exc
        path = Path(database).resolve(strict=True)
        self.api = cnl
        self.connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        self.connection.execute('PRAGMA query_only=ON')
        self.cursor = self.connection.cursor()
        self.exists = lru_cache(maxsize=65536)(self._exists)
        # Fail explicitly on the wrong SQLite distribution/schema.
        try:
            self.cursor.execute('SELECT id FROM concepts LIMIT 0')
            self.cursor.execute('SELECT id,start,end,rel,weight FROM relations LIMIT 0')
        except Exception:
            self.close()
            raise

    def _exists(self, uri: str) -> bool:
        return bool(self.api.does_concept_exist(concept_id=uri, db_cursor=self.cursor))

    def map_word(self, word: Word) -> tuple[str | None, str]:
        exact = concept_uri(word.surface)
        if self.exists(exact):
            return exact, 'exact'
        if word.lemma:
            lemma = concept_uri(word.lemma)
            if lemma != exact and self.exists(lemma):
                return lemma, 'lemma'
        return None, 'unmapped'

    def induced_assertions(self, concepts: set[str], relations: tuple[str, ...] = ()) -> list[tuple]:
        """Every direct assertion once, original orientation, no inferred edges.

        Upstream get_all_edges returns incoming AND outgoing relations. Each
        assertion is accepted only while visiting its source, eliminating double
        discovery. Multiple relation types/assertion IDs remain distinct.
        """
        allowed = {r if r.startswith('/r/') else '/r/' + r for r in relations}
        found = {}
        for uri in sorted(concepts):
            for edge in self.api.get_all_edges(cn_id=uri, db_cursor=self.cursor):
                if edge.start != uri or edge.end not in concepts:
                    continue
                if allowed and edge.rel not in allowed:
                    continue
                weight = float(edge.weight)
                if not math.isfinite(weight):
                    raise ValueError(f'Non-finite ConceptNet weight: {edge.id}')
                row = (edge.id, edge.start, edge.end, edge.rel, weight)
                if edge.id in found and found[edge.id] != row:
                    raise ValueError(f'Conflicting duplicate assertion ID: {edge.id}')
                found[edge.id] = row
        return [found[key] for key in sorted(found)]

    def close(self):
        self.exists.cache_clear()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
