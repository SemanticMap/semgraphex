"""Text -> ConceptNet occurrence graphs -> Parquet, using DuckDB joins.

Run `python -m semmap_haken.text_corpus --help`. Dependencies are lazy so the
existing numerical core remains usable without the optional text installation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass, fields
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .text_conceptnet import open_conceptnet, words

SCHEMA_VERSION = 1
OCCURRENCE_SCHEMA = '''doc_id VARCHAR, occurrence_id BIGINT, surface VARCHAR,
    lemma VARCHAR, char_start BIGINT, char_end BIGINT, concept_uri VARCHAR,
    mapping_status VARCHAR'''
ASSERTION_SCHEMA = '''assertion_id VARCHAR, source_concept VARCHAR,
    target_concept VARCHAR, relation_uri VARCHAR, conceptnet_weight DOUBLE'''


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    language: str = 'en'
    topic: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class Options:
    database: str
    output: str
    workers: int = 1
    threads_per_worker: int = 1
    memory_limit: str = '1GB'
    max_edges_per_document: int | None = 20_000_000
    relations: tuple[str, ...] = ()
    spacy_model: str | None = None
    database_version: str = 'unspecified; see database_sha256'

    def validate(self):
        if sys.version_info < (3, 12):
            raise RuntimeError('The text pipeline requires Python >=3.12 (ConceptNet Local)')
        if any(type(v) is not int for v in (self.workers, self.threads_per_worker)):
            raise ValueError('Worker and thread counts must be integers')
        if self.workers < 1 or self.threads_per_worker < 1:
            raise ValueError('Worker and thread counts must be positive')
        if self.max_edges_per_document is not None and (
                type(self.max_edges_per_document) is not int or self.max_edges_per_document < 1):
            raise ValueError('max_edges_per_document must be positive or null')
        if not Path(self.database).is_file():
            raise FileNotFoundError(self.database)
        if Path(self.output).resolve() == Path(self.database).resolve():
            raise ValueError('Output must be a directory, not the source database')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value):
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _connect(options: Options | None = None, scratch: Path | None = None):
    import duckdb
    connection = duckdb.connect(':memory:')
    connection.execute(f'SET threads={options.threads_per_worker if options else 1}')
    if options:
        connection.execute('SET memory_limit=' + _literal(options.memory_limit))
    if scratch:
        connection.execute('SET temp_directory=' + _literal(str(scratch)))
    return connection


def _copy(connection, query: str, destination: Path):
    connection.execute(f'COPY ({query}) TO {_literal(str(destination))} '
                       '(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)')


def read_documents(path: str | Path, language: str = 'en') -> list[Document]:
    source = Path(path).resolve(strict=True)
    documents = []
    if source.is_dir():
        for item in sorted(source.rglob('*.txt')):
            documents.append(Document(item.relative_to(source).as_posix(),
                                      item.read_bytes().decode('utf-8'), language,
                                      source=str(item)))
    elif source.suffix.lower() == '.txt':
        documents.append(Document(source.name, source.read_bytes().decode('utf-8'),
                                  language, source=str(source)))
    elif source.suffix.lower() == '.jsonl':
        for line_number, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if ('text' in row) == ('path' in row):
                raise ValueError(f'JSONL line {line_number}: provide exactly one of text/path')
            text_path = (source.parent / row['path']).resolve() if 'path' in row else None
            text = text_path.read_bytes().decode('utf-8') if text_path else row['text']
            documents.append(Document(row['doc_id'], text, row.get('language', language),
                                      row.get('topic'), str(text_path or source)))
    else:
        raise ValueError('Input must be a .txt, .jsonl, or directory of .txt files')
    _validate_documents(documents)
    return documents


def _validate_documents(documents):
    if not documents:
        raise ValueError('No input documents')
    seen = set()
    for doc in documents:
        if not isinstance(doc.doc_id, str) or not doc.doc_id or doc.doc_id in seen:
            raise ValueError('Document IDs must be nonempty unique strings')
        if not isinstance(doc.text, str):
            raise ValueError(f'{doc.doc_id}: text must be a string')
        if doc.language != 'en':
            raise ValueError(f'{doc.doc_id}: ConceptNet Local supports English only (en)')
        if doc.topic is not None and not isinstance(doc.topic, str):
            raise ValueError('topic must be a string or null')
        seen.add(doc.doc_id)


def _identity(doc: Document):
    return {'doc_id': doc.doc_id, 'language': doc.language, 'topic': doc.topic,
            'text_sha256': hashlib.sha256(doc.text.encode('utf-8')).hexdigest(),
            'key': hashlib.sha256(doc.doc_id.encode('utf-8')).hexdigest()}


def _fingerprint(options, documents):
    semantic = {k: v for k, v in asdict(options).items()
                if k not in {'output', 'database', 'workers', 'threads_per_worker', 'memory_limit'}}
    # A copied database is the same input; a changed DB/version/config is not.
    packages = {p: version(p) for p in ('conceptnet-local', 'duckdb')}
    if options.spacy_model:
        packages['spacy'] = version('spacy')
        packages[options.spacy_model] = version(options.spacy_model)
    from . import text_conceptnet
    return {'schema_version': SCHEMA_VERSION, 'options': semantic,
            'database_sha256': sha256(Path(options.database)), 'packages': packages,
            'code_sha256': {Path(p).name: sha256(Path(p))
                            for p in (__file__, text_conceptnet.__file__)},
            'documents': [_identity(d) for d in sorted(documents, key=lambda d: d.doc_id)]}


def _verify_document(directory: Path, identity: dict):
    if not (directory / 'COMPLETED').is_file():
        raise ValueError(f'Incomplete document output: {directory}')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['identity'] != identity:
        raise ValueError(f'Document identity mismatch: {directory}')
    for name, checksum in manifest['checksums'].items():
        if sha256(directory / name) != checksum:
            raise ValueError(f'Checksum mismatch: {directory / name}')
    return manifest


def _worker_setup(threads):
    # Executed before importing NumPy/ConceptNet Local inside spawned workers.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = str(threads)


def _build_document(task):
    doc, options, resume, status_dir = task
    identity = _identity(doc)
    final = Path(options.output) / 'documents' / identity['key']
    last_status = {'percent': -1, 'stage': None, 'timestamp': 0.0}

    def progress(stage, percent, current=None, total=None):
        """Publish a stage-based estimate; exact counts when available.

        Each worker writes only its own small JSON file on the LOCAL SSD.
        The coordinator polls these files and may mirror summaries to Drive.
        Do not block the worker on Google Drive I/O.
        """
        if status_dir is None:
            return
        percent = max(0, min(int(percent), 99))  # Drive publish marks 100%
        now = time.monotonic()
        if (stage == last_status['stage'] and percent == last_status['percent']
                and now - last_status['timestamp'] < 5):
            return
        status = {'doc_id': doc.doc_id, 'stage': stage, 'percent': percent,
                  'percent_kind': 'stage_estimate',
                  'current': current, 'total': total, 'updated_at': time.time()}
        _json(Path(status_dir) / (identity['key'] + '.json'), status)
        last_status.update({'stage': stage, 'percent': percent, 'timestamp': now})

    if final.exists():
        if not resume:
            raise FileExistsError(final)
        report = _verify_document(final, identity)
        progress('already_completed', 99)
        return report
    staging = Path(tempfile.mkdtemp(prefix='.pending-', dir=final.parent))
    started = time.perf_counter()
    db = None
    try:
        progress('initializing', 0)
        nlp = None
        if options.spacy_model:
            import spacy
            nlp = spacy.load(options.spacy_model, disable=['ner', 'parser'])
        progress('tokenizing', 0, 0, len(doc.text))
        tokens = words(
            doc.text, nlp,
            on_progress=lambda done, total: progress(
                'tokenizing', int(20 * done / max(total, 1)), done, total)
            if status_dir is not None else None,
        )
        progress('mapping_concepts', 20, 0, len(tokens))
        with open_conceptnet(options.database, doc.language) as cn:
            occurrences = []
            interval = max(1, len(tokens) // 100)
            for index, word in enumerate(tokens):
                uri, status = cn.map_word(word)
                assert doc.text[word.start:word.end] == word.surface
                occurrences.append((doc.doc_id, index, word.surface, word.lemma,
                                    word.start, word.end, uri, status))
                if index % interval == 0 or index + 1 == len(tokens):
                    progress('mapping_concepts',
                             20 + int(40 * (index + 1) / max(len(tokens), 1)),
                             index + 1, len(tokens))
            concepts = {row[6] for row in occurrences if row[6] is not None}
            progress('querying_relations', 60, 0, len(concepts))
            assertions = cn.induced_assertions(concepts, options.relations)
            progress('querying_relations', 68, len(concepts), len(concepts))
        progress('preparing_tables', 69)
        db = _connect(options, staging / 'spill')
        db.execute('CREATE TABLE occurrences (' + OCCURRENCE_SCHEMA + ')')
        db.execute('CREATE TABLE assertions (' + ASSERTION_SCHEMA + ')')
        if occurrences:
            db.executemany('INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?)', occurrences)
        if assertions:
            db.executemany('INSERT INTO assertions VALUES (?,?,?,?,?)', assertions)
        progress('counting_edges', 73)
        # Count the Cartesian expansion using concept frequencies, not expanded edges.
        expected = int(db.execute('''
            WITH counts AS (SELECT concept_uri, count(*) AS n FROM occurrences
                            WHERE concept_uri IS NOT NULL GROUP BY concept_uri)
            SELECT coalesce(sum(s.n*t.n - CASE WHEN a.source_concept=a.target_concept
                                              THEN s.n ELSE 0 END),0)
            FROM assertions a JOIN counts s ON s.concept_uri=a.source_concept
                              JOIN counts t ON t.concept_uri=a.target_concept
        ''').fetchone()[0])
        if options.max_edges_per_document is not None and expected > options.max_edges_per_document:
            raise ValueError(f'{doc.doc_id}: {expected} occurrence edges exceed '
                             f'max_edges_per_document={options.max_edges_per_document}; '
                             'nothing truncated; raise the cap explicitly')
        progress('writing_occurrences', 77, 0, len(tokens))
        _copy(db, 'SELECT * FROM occurrences ORDER BY occurrence_id', staging / 'occurrences.parquet')
        progress('writing_assertions', 81, 0, len(assertions))
        _copy(db, 'SELECT * FROM assertions ORDER BY assertion_id', staging / 'assertions.parquet')
        progress('writing_edges', 85, 0, expected)
        _copy(db, '''SELECT s.doc_id, s.occurrence_id AS source_occurrence_id,
                    t.occurrence_id AS target_occurrence_id, a.*,
                    abs(t.char_start-s.char_start) AS distance_chars,
                    t.char_start-s.char_start AS signed_offset,
                    1.0/(1+abs(t.char_start-s.char_start)) AS proximity_weight
             FROM assertions a
             JOIN occurrences s ON s.concept_uri=a.source_concept
             JOIN occurrences t ON t.concept_uri=a.target_concept
             WHERE s.occurrence_id <> t.occurrence_id
             ORDER BY source_occurrence_id, target_occurrence_id, assertion_id''',
              staging / 'edges.parquet')
        progress('verifying_edges', 94, expected, expected)
        edges_path = _literal(str(staging / 'edges.parquet'))
        actual = db.execute(f'SELECT count(*) FROM read_parquet({edges_path})').fetchone()[0]
        if actual != expected:
            raise RuntimeError('Occurrence expansion count mismatch')
        progress('writing_concept_edges', 95)
        _copy(db, f'''SELECT doc_id,source_concept,target_concept,relation_uri,
                   count(*) AS occurrence_pair_count,
                   count(DISTINCT assertion_id) AS assertion_count,
                   min(distance_chars) AS min_distance_chars,
                   avg(distance_chars) AS mean_distance_chars,
                   max(distance_chars) AS max_distance_chars,
                   sum(proximity_weight) AS proximity_weight_sum
            FROM read_parquet({edges_path})
            GROUP BY doc_id,source_concept,target_concept,relation_uri
            ORDER BY source_concept,target_concept,relation_uri''', staging / 'concept_edges.parquet')
        progress('finalizing', 97)
        (staging / 'text.txt').write_bytes(doc.text.encode('utf-8'))
        mapped = sum(row[6] is not None for row in occurrences)
        result = {'identity': identity, 'source': doc.source, 'word_count': len(tokens),
                  'mapped_words': mapped, 'unmapped_words': len(tokens)-mapped,
                  'coverage': mapped / len(tokens) if tokens else 0.0,
                  'concept_count': len(concepts), 'assertion_count': len(assertions),
                  'edge_count': expected, 'elapsed_seconds': time.perf_counter()-started,
                  'checksums': {p.name: sha256(p) for p in sorted(staging.glob('*.parquet'))}}
        result['checksums']['text.txt'] = sha256(staging / 'text.txt')
        db.close()
        db = None
        shutil.rmtree(staging / 'spill', ignore_errors=True)
        _json(staging / 'manifest.json', result)
        (staging / 'COMPLETED').write_text('complete\n', encoding='utf-8')
        os.replace(staging, final)
        progress('ready_for_checkpoint', 99)
        return result
    except BaseException:
        progress('failed', last_status['percent'])
        raise
    finally:
        if db is not None:
            db.close()
        if staging.exists():
            shutil.rmtree(staging)


def build_corpus(documents: list[Document], options: Options, *, resume=False,
                 on_document_completed=None, on_document_progress=None):
    """One coordinator, independent spawn workers, immutable per-document outputs."""
    options.validate()
    _validate_documents(documents)
    with open_conceptnet(options.database):
        pass  # Fail on missing dependencies/resources/schema before creating run output.
    options = Options(**{**asdict(options), 'database': str(Path(options.database).resolve()),
                         'output': str(Path(options.output).resolve())})
    database = Path(options.database)
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Use a frozen/checkpointed SQLite snapshot, not an active WAL database')
    source_stat = (database.stat().st_size, database.stat().st_mtime_ns)
    root = Path(options.output)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.writer.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode('ascii'))
    os.close(descriptor)
    try:
        contract = _fingerprint(options, documents)
        contract = json.loads(json.dumps(contract))  # normalize tuples for RESUME comparison
        run_path = root / 'run.json'
        if run_path.exists():
            if not resume:
                raise FileExistsError('Output exists; use --resume or a new output directory')
            if json.loads(run_path.read_text(encoding='utf-8'))['contract'] != contract:
                raise ValueError('RESUME contract mismatch: inputs/database/config/code changed')
        else:
            if any(p.name != '.writer.lock' for p in root.iterdir()):
                raise ValueError('Refusing to reuse a nonempty output without run.json')
            try:
                revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                    cwd=Path(__file__).parent, stderr=subprocess.DEVNULL, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = None
            _json(run_path, {'contract': contract, 'resolved_options': asdict(options),
                             'git_commit': revision, 'python': sys.version,
                             'distance_definition': 'abs(target.char_start-source.char_start)',
                             'offset_unit': 'Unicode code point, zero-based, original UTF-8 text',
                             'provenance': 'SQLite ConceptNet Local or HF normalized. The HF database has '
                                           'no original assertion ID, dataset, sources or license; '
                                           'hf55:edge_norm:<rowid> is synthetic, stable only within '
                                           'the pinned unchanged database snapshot.',
                             'interpretation': 'Co-occurring concepts connected in ConceptNet; '
                                               'not relations asserted by the text'})
        (root / 'COMPLETED').unlink(missing_ok=True)
        (root / 'documents').mkdir(exist_ok=True)
        # These tiny files are runtime observations, NOT durable checkpoints.
        # Only start this directory after run.json was validated/written.
        status_dir = root / '.document-progress' if on_document_progress else None
        if status_dir is not None:
            if status_dir.exists():
                shutil.rmtree(status_dir)
            status_dir.mkdir()
        tasks = [(doc, options, resume, str(status_dir) if status_dir else None)
                 for doc in sorted(documents, key=lambda d: d.doc_id)]
        reports = []
        if options.workers == 1 and on_document_progress is None:
            # Preserve cheap local sequential mode for callers not monitoring.
            for task in tasks:
                report = _build_document(task)
                if on_document_completed is not None:
                    on_document_completed(report)
                reports.append(report)
        else:
            # Even for one worker, a subprocess lets the coordinator poll real
            # document progress during long spaCy/SQL phases.
            with ProcessPoolExecutor(max_workers=min(options.workers, len(tasks)),
                                     mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_worker_setup,
                                     initargs=(options.threads_per_worker,)) as pool:
                futures = {pool.submit(_build_document, task): task[0].doc_id
                           for task in tasks}
                pending = set(futures)
                last_status = {}

                def poll_progress():
                    if status_dir is None:
                        return
                    for path in status_dir.glob('*.json'):
                        try:
                            status = json.loads(path.read_text(encoding='utf-8'))
                        except (ValueError, OSError):
                            continue
                        doc_id = status.get('doc_id')
                        snapshot = (status.get('percent'), status.get('stage'),
                                    status.get('current'), status.get('total'))
                        if doc_id and last_status.get(doc_id) != snapshot:
                            last_status[doc_id] = snapshot
                            on_document_progress(status)
                    # Periodic heartbeat also during a long single DuckDB query:
                    # a stage may stay at 85% while it is computing.
                    on_document_progress(None)

                while pending:
                    done, pending = wait(pending, timeout=2.0,
                                         return_when=FIRST_COMPLETED)
                    poll_progress()
                    for future in done:
                        report = future.result()
                        if on_document_completed is not None:
                            on_document_completed(report)
                        reports.append(report)
                poll_progress()
        reports.sort(key=lambda report: report['identity']['doc_id'])
        if (source_stat != (database.stat().st_size, database.stat().st_mtime_ns)
                or (wal.exists() and wal.stat().st_size)):
            raise ValueError('ConceptNet database changed during the run; corpus not published')
        manifest = {'schema_version': SCHEMA_VERSION, 'documents': reports,
                    'word_count': sum(r['word_count'] for r in reports),
                    'edge_count': sum(r['edge_count'] for r in reports),
                    'run_sha256': sha256(run_path)}
        _json(root / 'manifest.json', manifest)
        (root / 'COMPLETED').write_text('complete\n', encoding='utf-8')
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def open_corpus(directory: str | Path, *, verify: bool = True):
    """Return an in-memory DuckDB catalogue over only the published Parquet files.

    Paths resolve from the supplied root, so copying the completed corpus works.
    The caller owns/closes the connection. Verification checks SHA-256 by default.
    """
    root = Path(directory).resolve()
    if not (root / 'COMPLETED').is_file():
        raise ValueError('Corpus has no COMPLETED marker')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if sha256(root / 'run.json') != manifest['run_sha256']:
        raise ValueError('Corpus run metadata checksum mismatch')
    folders = []
    for row in manifest['documents']:
        folder = root / 'documents' / row['identity']['key']
        if verify and _verify_document(folder, row['identity']) != row:
            raise ValueError('Document manifest differs from the published corpus')
        folders.append(folder)
    db = _connect()
    try:
        for table in ('occurrences', 'edges', 'concept_edges'):
            paths = ','.join(_literal(str(folder / (table + '.parquet'))) for folder in folders)
            db.execute(f'CREATE VIEW {table} AS SELECT * FROM read_parquet([{paths}])')
        db.execute('CREATE TABLE documents (doc_id VARCHAR, language VARCHAR, topic VARCHAR, '
                   'word_count BIGINT, mapped_words BIGINT, edge_count BIGINT)')
        db.executemany('INSERT INTO documents VALUES (?,?,?,?,?,?)',
                       [(r['identity']['doc_id'], r['identity']['language'], r['identity']['topic'],
                         r['word_count'], r['mapped_words'], r['edge_count']) for r in manifest['documents']])
        return db
    except Exception:
        db.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--config', required=True, type=Path)
    build.add_argument('--resume', action='store_true')
    export = commands.add_parser('export')
    export.add_argument('--corpus', required=True, type=Path)
    export.add_argument('--doc-id', required=True)
    export.add_argument('--output', required=True, type=Path)
    export.add_argument('--level', choices=['occurrence', 'concept'], default='occurrence')
    export.add_argument('--undirected', action='store_true')
    export.add_argument('--max-edges', type=int, default=2_000_000)
    args = parser.parse_args(argv)
    if args.command == 'export':
        from .text_export import export_prepared
        result = export_prepared(args.corpus, args.doc_id, args.output, level=args.level,
                                 directed=not args.undirected, max_edges=args.max_edges)
    else:
        import yaml
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        if not isinstance(config, dict) or 'input' not in config:
            parser.error('Config must be a mapping with input/database/output')
        allowed = {f.name for f in fields(Options)} | {'input', 'language'}
        unknown = set(config) - allowed
        if unknown:
            parser.error(f'Unknown configuration keys: {sorted(unknown)}')
        parent = args.config.resolve().parent
        for key in ('input', 'database', 'output'):
            if key not in config:
                parser.error(f'Missing configuration key: {key}')
            config[key] = str((parent / config[key]).resolve())
        docs = read_documents(config.pop('input'), config.pop('language', 'en'))
        config['relations'] = tuple(config.get('relations') or ())
        result = build_corpus(docs, Options(**config), resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
