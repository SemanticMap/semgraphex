# Colab: ZIP → ConceptNet Local → DuckDB/Parquet → Google Drive

Notebook: `notebooks/10_text_corpus_conceptnet_local_drive.ipynb`.

## Google Drive contracts (fixed)

- Input ZIP ID: `1I8dYdI4eLVmlhZjNewxl63EGgJKLmDqR`, verified title `Txt.zip`, 244,637,041 bytes.
- Output folder ID: `1C7r8A0L0afrfq_8KTebOBM4BJqbZjXM4` (existing `colab` folder).
- Output experiment directory: `text-corpus-<EXPERIMENT_LABEL>` underneath that folder.
- Existing input is untouched. New label means a new isolated experiment.
- **ConceptNet Local SQLite database file is NOT in the supplied links.** The notebook requires its Google Drive file ID or explicit mounted path. It never downloads a random/incompatible DB.
- ConceptNet Local is English-only; Cyrillic-heavy corpora are stopped before graph construction, rather than silently labelled English.

## Instructions

1. Open the notebook through Colab, choose T4 or L4 GPU in Runtime settings (text extraction uses CPU; GPU only helps downstream).
2. Set `EXPERIMENT_LABEL` (safe ASCII characters). Keep it unchanged for future RESUME.
3. Set `CONCEPTNET_DB_DRIVE_FILE_ID` or `CONCEPTNET_DB_PATH` to a prebuilt **ConceptNet Local** SQLite database (schema: tables `concepts`, `relations`, required indexed neighbours).
4. Set `MAX_DOCS=100` for the first bounded pilot, or `None` to process all `.txt` members. Only UTF-8 `.txt` is ingested in this version. Other ZIP contents are not processed.
5. Run all cells. Authorize Google Drive. Text archive and SQLite are copied to Colab's local disk. The runner auto-selects worker count from CPUs/available memory.
6. Observe `progress.json` and per-document Parquet/manifest folders. After a disconnect simply reconnect and run all with the same label. Completed outputs are checked by SHA-256, restored, and skipped.

## Technical checkpoints

`build_corpus(..., on_document_completed=callback)` now reports each completed process task to the coordinator immediately using `as_completed`. The callback publishes a document's files to a temporary sibling folder on Drive, rechecks the data and manifest, then renames the complete folder. A separate `progress.json` is updated after each completed document. `COMPLETED` at corpus root is published only when all documents were verified.

Each run writes `experiment.json` with archive and SQLite SHA-256, selection policy and semantic settings. Changes to archive, SQLite, corpus selection or mapping settings for an existing label cause a clear error. A new label creates a new result without overwriting old experiments. `run.json` tracks package/code versions, and original `text_corpus` RESUME guards remain enabled.

The whole corpus must still fit within the reading strategy of `read_documents` (texts loaded in RAM). Runner enforces a default 4GB cap on total **selected uncompressed** TXT bytes, plus 64MB per TXT. Larger datasets require a streaming iteration rewrite, not an unreported truncation. On the example notebook `MAX_DOCS=100` intentionally selects the lexicographically first 100 TXT entries. Values beyond the configured maximum stop with errors (no inferred graph edges are dropped).

For full reproducibility the processed original ZIP is identified by its file SHA, not only its Drive ID, and Unicode offsets are preserved in the original text.

## Known limits

- Full 244MB ZIP file's content has not been inspected in this environment; Drive metadata verified, file bytes not downloaded.
- No production ConceptNet Local SQLite was provided or discovered, so the actual 20–100 document run cannot be claimed complete.
- Google Drive FUSE cannot guarantee the same atomic semantics as local POSIX disks. Files are copied to named staging directories and verified twice, but recovery validation is the authoritative correctness check.
- No claim that GPU accelerates SQL/ConceptNet matching; T4/L4 availability is diagnosed, not used in this stage.
- A document lost before its per-document checkpoint finishes is reprocessed on resume.
- If the notebook is run twice simultaneously with the same label, checkpoints may conflict. Run a single coordinator per experiment label.
- The original `text_corpus.py` interface does not support arbitrary multilingual ConceptNet URIs; an independent multilingual adapter is required for Russian texts.

## Validation

Execute `python -m pytest -q tests/test_text_corpus_words.py tests/test_text_corpus.py tests/test_text_colab.py`.

New tests cover ZIP order/traversal/size limits and checkpoint copying, restoration, SHA failure. Real full-corpus Colab run remains an independent test.
