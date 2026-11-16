## Длинные документы: последовательная обработка частями, единый граф

Начиная с ветки `feature/text-corpus-colab-drive-resume`, обновлённый Colab
использует `--chunked-graph`. Это решает ошибку
`23306031 occurrence edges exceed max_edges_per_document=20000000`
для документа `0000068_How_Scientists_Think_21_Experiments.txt` **без удаления рёбер**.

1. Исходный текст режется на фрагменты до `chunk_chars=100000`
   кодовых точек Unicode по границам слов. Фрагменты **последовательно**
   проходят spaCy и привязку ConceptNet; каждая порция слов добавляется
   в общую таблицу DuckDB `occurrences`. `occurrence_id` уникален во
   всём документе, `char_start/char_end` остаются абсолютными.
2. Только после получения всех вершин извлекаются все применимые
   отношения ConceptNet для **всего документа**.
3. Рёбра строятся по партиям **исходных** вхождений, но каждой партии
   доступны **все целевые слова документа**, в том числе из других
   текстовых фрагментов. Поэтому точный граф совпадает по вершинам,
   направлению, типам отношений, расстояниям и множеству рёбер
   с неразбитым построением.
4. При `edge_partition_rows=1000000` DuckDB пишет последовательно
   `edges_part_000000.parquet`, `edges_part_000001.parquet` и т.д.
   Каждый файл имеет не более миллиона рёбер. В редком случае, когда
   у одной исходной вершины рёбер больше лимита, дополнительно
   разделяются диапазоны *целевых* вхождений.
5. `open_corpus()` объединяет все части в один виртуальный
   DuckDB-view `edges`. Отчёт `concept_edges.parquet` агрегирует
   **все** рёбра, а экспорт `PreparedGraph` работает через тот же view.
   Обычные старые `edges.parquet` продолжают поддерживаться.

Флаги CLI для полного графа:

```bash
python -m semmap_haken.text_colab \
  --archive /content/drive/MyDrive/SemanticMap/Txt.zip \
  --hf-db --drive-parent /content/drive/MyDrive/SemanticMap/colab \
  --label semmap-text-hf55-spacy-chunked-01 \
  --chunked-graph --chunk-chars 100000 --edge-partition-rows 1000000 \
  --max-edges-per-document 0 --max-total-edges-per-document 150000000 \
  --max-docs 100 --spacy-model en_core_web_sm
```

`--max-edges-per-document 0` **отключает старый общий предел**
20 млн рёбер, только в режиме `--chunked-graph`. Новый
`max-total-edges-per-document` — независимый страховочный предел
для защиты диска; при превышении будет явная ошибка, а не усечение.
На промежуточных этапах прогресс `mapping_chunks` отображает реальное
число уже обработанных символов текста, а `writing_edges` —
реальное число рёбер, записанных в Parquet.

**Восстановление:** контрольные суммы и `COMPLETED` публикуются
на Google Drive после завершения **всего документа**, включая все
`edges_part_*.parquet`. Если Colab прервался *внутри* большого документа,
его незавершённые части локального SSD пересчитываются; завершённые и
опубликованные документы восстанавливаются полностью. Изменение
алгоритма/параметров требует **новой метки эксперимента**. Старые
каталоги результатов не удаляются. Для восстановления внутри
незавершённого документа потребовался бы отдельный устойчивый
checkpoint по частям, который пока не реализован.

## Прогресс выполнения (документы и проценты)

Во время вычислений с несколькими `--workers` координатор каждые ~10 секунд
печатает в Colab и сохраняет в Google Drive `progress.json` такую сводку:

```text
[progress] Документы: 12/100 (12.0%) | article-1.txt: 42% (поиск понятий ConceptNet, 42000/100000) | article-2.txt: 85% (запись рёбер в Parquet)
```

- `completed` / `total` и `completed_percent` — **точно** сколько
  документов обработано и опубликовано на Drive; завершённые документы после
  RESUME учитываются сразу.
- `active_documents` хранит отдельную запись каждого выполняющегося
  текста: `doc_id`, `percent` (0–99), `stage`, `stage_label`,
  `current` и `total` (когда доступны).
- **Процент отдельного документа является оценкой по этапам**, а не
  измеренной долей процессорного времени: обработка символов и сопоставление
  слов дают настоящие численные счётчики; длинные DuckDB запросы
  могут показывать неизменные 85% до окончания SQL-операции.
- Короткие временные статусы пишутся **локально** независимо от процессов,
  а на Google Drive передаются только агрегированные обновления,
  чтобы не перегружать Drive FUSE. Сигнал `100%` соответствует завершённой
  проверенной публикации документа, а не просто локально готовому Parquet.

Прогресс — наблюдение, а не механизм RESUME; восстановление по-прежнему
опирается на `run.json`, `COMPLETED` и проверенные SHA-256 файлов.

> **Исправление Colab spaCy E088 (15 ноября 2026):** диагностирован
> `ValueError: [E088]` при длине документа **4 366 226** символов,
> превышающей лимит `nlp.max_length=1 000 000`. Для `SPACY_MODEL`
> лемматизация теперь разбивает текст по границам пробелов/предложений
> на фрагменты до 100 000 символов; исходный текст, номера Unicode
> кодовых точек и все вхождения слов сохраняются. Нельзя просто
> повышать `nlp.max_length` при 7 процессах × 4 GB DuckDB, это риск OOM.
> Обновлённый ноутбук проверяет доступные CPU/RAM, безопасно ограничивает
> параллельность, печатает полный вывод подпроцесса и читает Drive
> `progress.json` после ошибки. Не нужно отключать spaCy.
>
> Наблюдалось `completed_this_session: 0` и отсутствие `run.json`
> в папке `semmap-text-hf55-spacy-fixed-01`; такой прерванный запуск
> поддерживает безопасную повторную инициализацию с той же меткой.
> Если другая сессия уже успела опубликовать документ и `run.json`,
> действуют обычные проверки контракта RESUME и несовместимые результаты
> не смешиваются.

> **Исправленная версия (рекомендуется):** [новый чистый Colab-ноутбук](https://colab.research.google.com/github/SemanticMap/semgraphex/blob/feature%2Ftext-corpus-colab-drive-resume/notebooks/11_text_corpus_hf55_colab_clean.ipynb) — без monkey-patch.
> Для нормализованного ConceptNet колонка `node_norm.language` теперь необязательна:
> язык определяется из URI `/c/en/...`; при этом таблицы `node_norm/rel_norm/edge_norm`
> и их действительно необходимые поля по-прежнему проверяются.
> При повторной попытке инициализации `progress.json` без `run.json` разрешён,
> если в папке ещё нет опубликованных документов. Данные без контракта не принимаются
> автоматически. Существующие папки экспериментов не удаляйте. Старый ноутбук 10
> содержал динамические патчи и **не предназначен для запуска**.

# Colab: Txt.zip → Hugging Face ConceptNet normalized → DuckDB/Parquet → Drive

Notebook: `notebooks/11_text_corpus_hf55_colab_clean.ipynb` (рекомендуется). 
Open it directly: https://colab.research.google.com/github/SemanticMap/semgraphex/blob/feature/text-corpus-colab-drive-resume/notebooks/11_text_corpus_hf55_colab_clean.ipynb

## Fixed sources and result folder

- TXT source: Google Drive `1I8dYdI4eLVmlhZjNewxl63EGgJKLmDqR`, `Txt.zip` (244,637,041 bytes, metadata inspected).
- Checkpoints/result parent: `1C7r8A0L0afrfq_8KTebOBM4BJqbZjXM4` (Google Drive `colab`).
- ConceptNet data source: https://huggingface.co/datasets/cstr/conceptnet-normalized-multi
- File: `conceptnet_normalized.db` (about 1.78 GB), **pinned** dataset revision `4d3fd17e80543e88d9823e2f4fe47901dcf9a4e6`.
- `huggingface_hub.hf_hub_download(repo_type="dataset", revision=..., cache_dir="/content/semmap-hf-cache")` downloads directly to Colab host. A cached snapshot is reused in the same live runtime, and the hub client supports caching/retry.
- Experiment directory: `text-corpus-<EXPERIMENT_LABEL>` under the given Drive parent.

## Important schema correction

`cstr/conceptnet-normalized-multi` is **not** the SQLite format for the Python package `conceptnet-local`:
- ConceptNet Local: `concepts`, `relations` with URI strings;
- Hugging Face normalized: `node_norm(node_pk,node_url,language)`, `rel_norm(rel_pk,rel_url)`, `edge_norm(start_fk,end_fk,rel_fk,weight)`.

`text_conceptnet.open_conceptnet()` detects the schema and chooses either the legacy adapter or `NormalizedConceptNet`. The HF adapter resolves URLs to integer node IDs, uses a small process-local TEMP table of concepts and a join against `edge_norm`, retains edge direction, relation and the source weights. It never writes to the HF SQLite file (all main DB connections use `mode=ro`).

The HF table does **not** expose the original edge assertion identifier or ConceptNet attribution fields (`dataset/sources/license`). Output `assertion_id` for HF is a **synthetic** identifier `hf55:edge_norm:<SQLite rowid>`; rowid is meaningful only for the immutable downloaded snapshot (recorded as SHA-256). Document graphs must not claim that it is an original assertion ID.

This specific dataset filters edges to languages `en/fr/it/de/es/ar/fa/grc/he/la/hbo` (and some cross-language edges) — **no full Russian-to-Russian graph**. The current text matcher still handles **English only**, and fails early on documents containing substantial Cyrillic.

## Run from Colab

1. Select Runtime T4 or L4 and Python 3.12+. (The text-construction phase uses CPU; GPU accelerates only later graph algorithms.)
2. In the notebook config cell, choose an experiment label. `MAX_DOCS=100` runs a bounded pilot of first 100 TXT files; `None` processes all TXT files in the archive.
3. Run all. Authenticate Google Drive when prompted. The notebook:
   - resolves the provided Drive IDs to mounted file/folder paths;
   - checks Colab CPU/RAM/SSD/GPU;
   - fetches latest branch code and installs `.[text]`, including `huggingface_hub`;
   - downloads the pinned normalized SQLite file to **local** `/content/semmap-hf-cache`;
   - validates tables/columns and reads the English node sample;
   - runs `semmap_haken.text_colab` with `--database` and an explicit dataset revision label.
4. Watch `text-corpus-<label>/progress.json` on Drive. After an interruption, **Run all with the same label**. All previously published document checkpoints are hashed and restored. No manual SQLite path is needed.

A headless CLI alternative, if mounting Drive already done:

```bash
python -m semmap_haken.text_colab \
  --archive "/content/drive/MyDrive/<path>/Txt.zip" \
  --hf-db \
  --drive-parent "/content/drive/MyDrive/<output-parent>" \
  --label "semmap-text-hf55-01" \
  --workers 4
```

Use `--hf-revision <SHA>` only to intentionally change dataset version. An existing label will reject changes in SQLite contents or version; use a **new label** for new data. For local testing, `--database <path>` remains available.

## Автоматическое подключение spaCy

В первой кодовой ячейке ноутбука установлено `SPACY_MODEL = "en_core_web_sm"`.
В отдельной ячейке **после установки semgraphex и перед скачиванием данных/запуском**
выполняется `pip install 'spacy>=3.7,<4'`; если модель отсутствует, выполняется
`python -m spacy download en_core_web_sm`. Затем ноутбук проверяет
`nlp("Cats are running.")[0].lemma_ == "cat"` и передаёт модель в CLI через
`--spacy-model`. Больше ничего вручную вставлять не нужно.

Если лемматизация не нужна, укажите `SPACY_MODEL = None`, установка пропустится.
**При изменении SPACY_MODEL задайте новую EXPERIMENT_LABEL**: токенизация и
сопоставление понятий могут измениться, а RESUME не должен смешивать результаты.
Модель только английская, как и текущий адаптер данных.

## Durable checkpoints, parallelism and reproducibility

- Independent worker processes (`spawn`) each read the SQLite DB in `mode=ro`; CPU workers are capped using available RAM and CPU cores; DuckDB operates on local SSD with one thread per worker by default.
- The coordinator receives each finished task immediately (`as_completed`) and publishes the entire completed document to its Drive `documents/<sha256(doc_id)>` directory. Includes Parquet, source text, manifest and SHA-256; no silently truncated edges.
- `progress.json` is updated periodically (~10 seconds), plus after each completed document. `experiment.json` includes SHA-256 of ZIP and SQLite and selected TXT list policy; `run.json` includes code/package versions. Root `COMPLETED` appears only after all results pass verification.
- `RESUME` copies only valid remote document outputs back to the local scratch directory and skips them. A document interrupted before publication is reprocessed.
- **Important:** a new Colab VM generally loses the local Hugging Face cache and must download the ~1.78 GB SQLite again; document checkpoints on Drive persist. For near-instant startup across Colab sessions a separate, user-approved persistent cache of the DB would be needed.
- FUSE mount atomicity is not guaranteed. The runner stages and validates files before publishing markers; recovery validation is authoritative.
- ZIP parsing requires UTF-8 TXT and rejects directory traversal, malformed code points, overly large uncompressed files, and mixed unsupported-language inputs.

## Verification

```bash
python -m pip install -e '.[text,dev]'
python -m nltk.downloader stopwords
python -m pytest -q tests/test_text_corpus_words.py tests/test_text_corpus.py tests/test_text_colab.py tests/test_text_hf.py tests/test_text_spacy_chunks.py tests/test_text_chunk_graph.py
```

HF download tests mock `huggingface_hub.hf_hub_download` and use a tiny SQLite `node_norm/edge_norm/rel_norm` fixture. Full 1.78 GB database download and the real TXT corpus in Google Colab have **not** been executed in CI; this must be validated in a real user Colab runtime. Performance and coverage are not claimed without real benchmarks.

HF dataset card: https://huggingface.co/datasets/cstr/conceptnet-normalized-multi
Hub download API: https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
