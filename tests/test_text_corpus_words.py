"""Dependency-light contract tests; no database, DuckDB or network needed."""
from pathlib import Path
import pytest
from semmap_haken.text_conceptnet import Word, concept_uri, words
from semmap_haken.text_corpus import Document, read_documents, _validate_documents


def test_offsets_and_all_words():
    text = "Cat\tcat.\r\nA naïve cafe\u0301 — кот 😀 can't-stop"
    found = words(text)
    assert [w.surface for w in found] == ['Cat', 'cat', 'A', 'naïve', 'cafe\u0301', 'кот', "can't", 'stop']
    assert all(text[w.start:w.end] == w.surface for w in found)
    assert found[1].start - found[0].start == 4
    assert words('😀cat animal')[0].start == 1  # code points, not UTF-8 bytes
    assert concept_uri('CAFE\u0301') == '/c/en/café'


def test_no_words():
    assert words(' \t\r\n😀...') == []


def test_input_preserves_crlf(tmp_path):
    p = tmp_path / 'text.txt'
    p.write_bytes(b'cat\r\nanimal')
    assert read_documents(p)[0].text == 'cat\r\nanimal'


def test_input_manifest(tmp_path):
    (tmp_path / 'a.txt').write_text('cat', encoding='utf-8')
    manifest = tmp_path / 'docs.jsonl'
    manifest.write_text('{"doc_id":"x","path":"a.txt","topic":"animals"}\n', encoding='utf-8')
    doc = read_documents(manifest)[0]
    assert (doc.text, doc.topic) == ('cat', 'animals')


@pytest.mark.parametrize('docs', [[], [Document('a', 'x'), Document('a', 'y')],
                                  [Document('a', 'кот', 'ru')]])
def test_input_rejected(docs):
    with pytest.raises(ValueError):
        _validate_documents(docs)
