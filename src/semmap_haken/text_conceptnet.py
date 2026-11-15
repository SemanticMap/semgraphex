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

# The Hugging Face cstr/conceptnet-normalized-multi distribution is NOT a
# ConceptNet Local database: it contains node_norm/rel_norm/edge_norm rather
# than concepts/relations. Keep both adapters behind the same small interface.
_NORMALIZED_HOSTS = ("http://conceptnet.io", "https://conceptnet.io")


def _normal_uri(uri: str, *, prefix: str) -> str:
    if uri.startswith(prefix):
        return uri[len(prefix):]
    if uri.startswith("/") and prefix == "http://conceptnet.io":
        return uri
    raise ValueError(f"Unexpected ConceptNet URL: {uri!r}")


class NormalizedConceptNet:
    """Read-only adapter for cstr/conceptnet-normalized-multi (ConceptNet 5.5).

    The DB has no original assertion IDs or sources. For provenance we use the
    frozen SQLite rowid, prefixed hf55:edge_norm:; it is NOT a source assertion ID.
    Each process owns its connection. A private SQLite TEMP table bounds the
    number of SQL parameters and avoids an N×N Python pair loop.
    """

    def __init__(self, database: str | Path, language: str = "en"):
        if language != "en":
            raise ValueError("The text pipeline currently maps English (/c/en/) only; "
                             "Russian is not included in the HF normalized edge filter")
        path = Path(database).resolve(strict=True)
        self.connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        self.cursor = self.connection.cursor()
        try:
            for table, columns in (
                ("node_norm", {"node_pk", "node_url"}),
                ("rel_norm", {"rel_pk", "rel_url"}),
                ("edge_norm", {"start_fk", "end_fk", "rel_fk", "weight"}),
            ):
                found = {row[1] for row in self.connection.execute(
                    "PRAGMA table_info(" + table + ")")}
                if not columns <= found:
                    raise ValueError("Incompatible normalized SQLite schema: "
                                     f"{table} missing {sorted(columns - found)}")
            # Never create persistent indices in the HF cache. The source
            # distribution is an immutable input. TEMP is per-worker.
            self.connection.execute(
                "CREATE TEMP TABLE _selected_concepts ("
                "node_pk INTEGER PRIMARY KEY, uri TEXT NOT NULL)")
            self._lookup_fk = lru_cache(maxsize=65536)(self._lookup_fk_uncached)
        except Exception:
            self.connection.close()
            raise

    def _lookup_fk_uncached(self, concept: str) -> int | None:
        urls = tuple(host + concept for host in _NORMALIZED_HOSTS)
        row = self.cursor.execute(
            "SELECT node_pk FROM node_norm WHERE node_url IN (?,?) LIMIT 1",
            urls).fetchone()
        return None if row is None else int(row[0])

    def map_word(self, word: Word) -> tuple[str | None, str]:
        exact = concept_uri(word.surface)
        if self._lookup_fk(exact) is not None:
            return exact, "exact"
        if word.lemma:
            lemma = concept_uri(word.lemma)
            if lemma != exact and self._lookup_fk(lemma) is not None:
                return lemma, "lemma"
        return None, "unmapped"

    def induced_assertions(self, concepts: set[str],
                           relations: tuple[str, ...] = ()) -> list[tuple]:
        if not concepts:
            return []
        allowed = {r if r.startswith("/r/") else "/r/" + r
                   for r in relations}
        selected = []
        for uri in sorted(concepts):
            fk = self._lookup_fk(uri)
            if fk is not None:
                selected.append((fk, uri))
        if not selected:
            return []
        self.cursor.execute("DELETE FROM temp._selected_concepts")
        self.cursor.executemany(
            "INSERT INTO temp._selected_concepts (node_pk, uri) VALUES (?,?)",
            selected)
        rows = self.cursor.execute("""
            SELECT e.rowid, src.uri, dst.uri, r.rel_url, e.weight
            FROM edge_norm AS e
            JOIN temp._selected_concepts AS src ON src.node_pk = e.start_fk
            JOIN temp._selected_concepts AS dst ON dst.node_pk = e.end_fk
            JOIN rel_norm AS r ON r.rel_pk = e.rel_fk
            ORDER BY e.rowid
        """)
        result = []
        for rowid, src, dst, relation_url, weight in rows:
            if relation_url.startswith(_NORMALIZED_HOSTS):
                relation = relation_url[len(
                    "https://conceptnet.io" if relation_url.startswith("https:")
                    else "http://conceptnet.io"):]
            elif relation_url.startswith("/r/"):
                relation = relation_url
            else:
                raise ValueError(f"Unexpected relation URL {relation_url!r}")
            if allowed and relation not in allowed:
                continue
            value = float(weight)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite ConceptNet weight at SQLite rowid {rowid}")
            result.append((f"hf55:edge_norm:{rowid}", src, dst,
                           relation, value))
        return result

    def close(self):
        self._lookup_fk.cache_clear()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open_conceptnet(database: str | Path, language: str = "en"):
    """Select the schema-specific reader before starting the corpus job."""
    path = Path(database).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    if {"node_norm", "edge_norm", "rel_norm"} <= tables:
        return NormalizedConceptNet(path, language=language)
    if {"concepts", "relations"} <= tables:
        return ConceptNetLocal(path, language=language)
    raise ValueError("Unsupported ConceptNet SQLite schema: " + repr(sorted(tables)))
