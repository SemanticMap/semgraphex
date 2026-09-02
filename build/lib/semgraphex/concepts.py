from typing import List, Iterable, Dict, Tuple
from collections import Counter, defaultdict
import math
import spacy
from .preprocess import load_spacy, sentences, tokenize


class ConceptCandidate:
    def __init__(self, text: str, score: float):
        self.text = text
        self.score = score
    def __repr__(self):
        return f"ConceptCandidate(text={self.text!r}, score={self.score:.3f})"


def extract_ner_concepts(texts: Iterable[str], min_freq: int = 2) -> List[ConceptCandidate]:
    nlp = load_spacy()
    counts: Counter = Counter()
    for doc in nlp.pipe(texts, batch_size=32):
        for ent in doc.ents:
            label = ent.label_
            if label in {"PERSON","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE"}:
                key = ent.text.strip()
                counts[key] += 1
    return [ConceptCandidate(t, c) for t, c in counts.items() if c >= min_freq]


def extract_statistical_terms(texts: Iterable[str], top_k: int = 50) -> List[ConceptCandidate]:
    # Simple TF scoring across corpus sentences
    sents = sentences(texts)
    tf: Counter = Counter()
    for s in sents:
        toks = set(tokenize(s))  # set => approximate DF-like rarity emphasis
        for tok in toks:
            tf[tok] += 1
    if not tf:
        return []
    total = sum(tf.values())
    scored = [(w, f/total) for w, f in tf.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [ConceptCandidate(w, s) for w, s in scored[:top_k]]


def merge_concept_lists(*lists: List[ConceptCandidate], max_items: int = 200) -> List[ConceptCandidate]:
    best: Dict[str, float] = {}
    for lst in lists:
        for c in lst:
            best[c.text] = max(best.get(c.text, 0.0), c.score)
    items = [ConceptCandidate(t, s) for t, s in best.items()]
    items.sort(key=lambda c: c.score, reverse=True)
    return items[:max_items]


def collect_concept_contexts(texts: List[str], concepts: List[ConceptCandidate], window_sentences: int = 3) -> Dict[str, List[str]]:
    # Map each concept to a list of sentences where it appears (with small window merging)
    sents = sentences(texts)
    concept_set = {c.text.lower(): c for c in concepts}
    contexts: Dict[str, List[str]] = defaultdict(list)
    for i, s in enumerate(sents):
        low = s.lower()
        for c_text in concept_set:
            if c_text in low:
                # take window
                start = max(0, i - window_sentences)
                end = min(len(sents), i + window_sentences + 1)
                block = ' '.join(sents[start:end])
                contexts[c_text].append(block)
    return contexts
