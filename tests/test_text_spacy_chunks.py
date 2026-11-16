"""spaCy chunking must not hit E088 or change original character offsets."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from semmap_haken.text_conceptnet import _spacy_chunks, words


@dataclass(frozen=True)
class FakeToken:
    text: str
    idx: int
    lemma_: str


class LimitCheckingNLP:
    def __init__(self, max_length=80):
        self.max_length = max_length
        self.calls = []

    def __call__(self, chunk):
        assert len(chunk) < self.max_length
        self.calls.append(len(chunk))
        return [FakeToken(m.group(), m.start(), m.group().lower())
                for m in re.finditer(r"\w+", chunk)]


def test_multiple_chunks_retain_all_absolute_offsets_and_lemmas():
    nlp = LimitCheckingNLP(60)
    text = "😀 Cats run.\r\n" * 150 + "The ANIMAL is happy."
    result = words(text, nlp)
    expected = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\w+", text)]
    assert [(w.start, w.end, w.surface) for w in result] == expected
    assert result[0].lemma == "cats"
    assert all(text[w.start:w.end] == w.surface for w in result)
    assert len(nlp.calls) > 5


def test_long_novel_above_spacy_default_limit_does_not_hit_e088():
    text = "Cats run quickly.\n" * 240_000
    nlp = LimitCheckingNLP(1_000_000)
    boundaries = list(_spacy_chunks(text, nlp))
    assert len(text) > 4_000_000
    assert 40 <= len(boundaries) <= 100
    assert all(len(chunk) < nlp.max_length for _, chunk in boundaries)
    assert "".join(chunk for _, chunk in boundaries) == text
    assert all(i == 0 or text[i - 1].isspace() for i, _ in boundaries)


def test_no_whitespace_run_fails_explicitly_without_silent_truncation():
    nlp = LimitCheckingNLP(50)
    with pytest.raises(ValueError, match="no whitespace"):
        words("A" * 100, nlp)


def test_without_spacy_behavior_unchanged():
    text = "Cats run.\r\nAn animal."
    assert [(w.start, w.surface) for w in words(text)] == [
        (0, "Cats"), (5, "run"), (11, "An"), (14, "animal")
    ]


def test_chunk_size_respects_override_max_length():
    nlp = LimitCheckingNLP(31)
    text = "Cat animal. " * 200
    output = words(text, nlp)
    assert len(output) == 400
    assert max(nlp.calls) <= 30
