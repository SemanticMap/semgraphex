import re
from typing import List, Iterable
import spacy
from spacy.language import Language
from importlib import util as _import_util
from nltk.corpus import stopwords
import nltk

# Ensure stopwords downloaded (lightweight check)
try:
    _ = stopwords.words('english')
except LookupError:  # pragma: no cover - runtime setup
    nltk.download('stopwords')

_URL_RE = re.compile(r'https?://\S+')
_NON_WORD = re.compile(r"[^a-zA-Z0-9']+")

_nlp: Language = None

def load_spacy(model: str = 'en_core_web_sm') -> Language:
    """Load spaCy model, attempting download if missing.

    This avoids failing at runtime in fresh environments.
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(model, disable=["parser"])  # speed
        except OSError:  # model not installed
            try:
                from spacy.cli import download
                download(model)
                _nlp = spacy.load(model, disable=["parser"])  # retry
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Failed to load spaCy model '{model}': {e}")
        if 'sentencizer' not in _nlp.pipe_names:
            _nlp.add_pipe('sentencizer')
    return _nlp


def normalize(text: str) -> str:
    text = _URL_RE.sub(' ', text)
    text = text.lower()
    return text


def sentences(texts: Iterable[str]) -> List[str]:
    nlp = load_spacy()
    sents = []
    for doc in nlp.pipe((normalize(t) for t in texts), batch_size=32):
        sents.extend([s.text.strip() for s in doc.sents if s.text.strip()])
    return sents


def tokenize(sent: str, remove_stop=True, lemma=True) -> List[str]:
    nlp = load_spacy()
    doc = nlp(sent)
    sw = set(stopwords.words('english')) if remove_stop else set()
    toks = []
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        if remove_stop and t.text.lower() in sw:
            continue
        form = t.lemma_.lower() if lemma else t.text.lower()
        form = _NON_WORD.sub('', form)
        if not form:
            continue
        toks.append(form)
    return toks
