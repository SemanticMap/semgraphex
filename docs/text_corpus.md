# Корпус текстов → графы вхождений ConceptNet

> **Обновление для Colab (Hugging Face):** готовая база скачивается автоматически
> из [cstr/conceptnet-normalized-multi](https://huggingface.co/datasets/cstr/conceptnet-normalized-multi)
> в `/content/semmap-hf-cache` через `huggingface_hub`.
> Этой базе нужна схема `node_norm/rel_norm/edge_norm`, а не `concepts/relations`;
> `open_conceptnet()` автоматически выбирает соответствующий адаптер.
> Описание ручной базы ConceptNet Local ниже оставлено для совместимости.
> Полный запуск в Colab описан в [text_colab_drive.md](text_colab_drive.md).
> Исходный фильтр Hugging Face исключает полный русский граф.


Основа: `bb91cbdbbd53ae48d64bd048aace8b94c0d186d1`, ветка
`feature/nsbm-multiscale-self-similarity` (последние изменения разработки, не `main`).
Изменения изолированы от существующих научных экспериментов.

## Контракт

Вход: папка UTF-8 `.txt`, один `.txt` или JSONL-манифест. Для каждого текста
создаётся отдельный граф. Основная вершина — **вхождение слова**, а не уникальное
понятие. Каждое слово сохраняется, в том числе стоп-слова, неизвестные слова и
изолированные вершины. Неизвестным словам присваивается `unmapped`, связи не
выдумываются. Только английский: это ограничение выбранной библиотеки и её базы.
`language: ru` и другие явно указанные языки отклоняются до обработки.
Автоматическое определение языка не выполняется.

Без дополнительных моделей поиск — точное совпадение нормализованного слова
(NFC + lower) с полным URI. Это не определение смысла слова по контексту.
Опциональный spaCy даёт лемму как запасной вариант после точного совпадения;
он также задаёт свою токенизацию. Senses, синонимы и FormOf не склеиваются.
Словосочетания в этой версии отдельно не выделяются.

Для каждого прямого утверждения ConceptNet `A → B` соединяются **все** вхождения
A со **всеми** вхождениями B в том же документе. Направление и разные отношения
сохраняются; одно утверждение не дублируется при обнаружении с обоих концов.
Само вхождение с собой не соединяется. Разные вхождения одного понятия могут
соединяться, только если соответствующее утверждение имеется в источнике.
Нет окна, ближайшего соседа, top-k или рёбер между разными документами.

```
distance_chars = abs(target.char_start - source.char_start)
signed_offset  = target.char_start - source.char_start
```

Координаты — кодовые точки Unicode с нуля; полуинтервалы `[start,end)`.
Пробелы, пунктуация, табуляция, CR и LF учитываются. Исходный текст не меняется.
Расстояние не используется как сила связи. Отдельная производная величина:
`proximity_weight = 1/(1+distance_chars)`. Вес ConceptNet сохраняется отдельно.

**Интерпретация:** слова присутствуют в тексте и связаны в ConceptNet. Это не
доказательство, что текст утверждает данное отношение, и не анализ отрицаний.

## Установка и запуск

Нужен Python **3.12+**. Базовый численный пакет сохраняет поддержку Python 3.10+;
новые зависимости изолированы в extra `text`.

```bash
python -m pip install -e '.[text]'
python -m nltk.downloader stopwords
# Разместите совместимую ConceptNet Local SQLite-базу в data/cn.db.
# Загрузка многогигабайтной базы не выполняется автоматически.
semmap-text-corpus build --config configs/text_corpus.yaml
# После прерывания, с теми же входами, кодом и настройками:
semmap-text-corpus build --config configs/text_corpus.yaml --resume
```

NLTK stopwords нужен при импорте самой библиотеки; наш алгоритм эти слова **не
удаляет**. Никаких загрузок ресурсов в процессе построения нет.

Для лемматизации:

```bash
python -m pip install -e '.[text,text-lemma]'
python -m spacy download en_core_web_sm
# В YAML: spacy_model: en_core_web_sm
```

Версию модели необходимо закрепить для публикационного запуска. Её фактическая
версия входит в контракт воспроизводимости. Смена модели/токенизации требует
нового каталога результата.

Пример JSONL (одна строка — один документ):

```jsonl
{"doc_id":"physics-01","path":"texts/physics.txt","language":"en","topic":"physics"}
{"doc_id":"biology-01","text":"A cat is an animal.","language":"en","topic":"biology"}
```

`path` разрешается относительно JSONL; пути `input`, `database`, `output` в YAML —
относительно самого YAML, не рабочего каталога. ID документов уникальны.
Каталоги документов названы SHA-256 от ID, поэтому `/`, кавычки и Unicode в ID
не превращаются в пути файловой системы.

## Хранение и запросы

```
run/
  run.json                 # хеш SQLite, код, версии пакетов, настройки, входы
  manifest.json            # опубликованный список документов и статистика
  COMPLETED
  documents/<sha256(doc_id)>/
    text.txt
    occurrences.parquet    # все слова, исходные позиции, URI, mapping_status
    assertions.parquet     # применимые утверждения id/start/end/rel/weight
    edges.parquet          # все пары вхождений и расстояния
    concept_edges.parquet  # агрегаты по понятиям и отношениям
    manifest.json          # coverage, счётчики, SHA-256 файлов
    COMPLETED
```

ConceptNet Local не возвращает `dataset`, `sources`, `license` через выбранный
API. Эта потеря источников явно записана в `run.json`; она не замаскирована
вымышленными метаданными. Использование официального полного дампа с такими
полями потребует другого адаптера. База идентифицируется полным SHA-256, а её
объявленную версию задаёт пользователь (`database_version`).

```python
from semmap_haken.text_corpus import open_corpus

con = open_corpus('tmp/runs/text-corpus-01')
try:
    print(con.execute('''
        SELECT d.topic, e.relation_uri, count(*) AS pairs,
               avg(e.distance_chars) AS mean_distance
        FROM edges e JOIN documents d USING (doc_id)
        GROUP BY d.topic, e.relation_uri
        ORDER BY d.topic, pairs DESC
    ''').fetchall())
finally:
    con.close()
```

Это DuckDB-каталог в памяти поверх Parquet, без общего записываемого `.duckdb`.
Читаются только файлы из опубликованного манифеста, не временные файлы.
Перенос всего каталога результата не ломает пути. Проверка SHA-256 включена по
умолчанию; `verify=False` отключает повторное полное чтение файлов для доверенных
неизменяемых результатов, но не заменяет проверку при приёмке.

## Подключение к semgraphex

```bash
semmap-text-corpus export --corpus tmp/runs/text-corpus-01 \
  --doc-id animals-01 --level occurrence --undirected \
  --output tmp/prepared/animals-occurrences

semmap-text-corpus export --corpus tmp/runs/text-corpus-01 \
  --doc-id animals-01 --level concept --undirected \
  --output tmp/prepared/animals-concepts
```

Экспорт использует существующие `PreparedGraph` и `save_prepared_graph()`:
`adjacency.npz`, `nodes.json`, `selected_edges.jsonl`, `graph_metadata.json`,
`resolved_config.json`, `COMPLETED`. Копируются также текст и таблица вхождений.
Идентификаторы концов `selected_edges` согласованы с `nodes.json`, поэтому
`relation_layers_from_prepared()` восстанавливает типизированные слои.
Для численной смежности суммируются proximity weights, а не расстояния и не
вероятности ConceptNet. Симметризация происходит только с `--undirected`.

Затем результат читается штатно:

```python
from semmap_haken.graph_build import load_prepared_graph
prepared = load_prepared_graph('tmp/prepared/animals-occurrences')
```

Существующим процедурам с предпосылкой связности нужно явно выбрать компоненту
в отдельном эксперименте; исходный экспорт изолированные вершины не удаляет.
Конечный граф текста сам по себе не является оценённым графоном/графексом.

## Производительность и восстановление

Каждый spawned-процесс обрабатывает документ независимо и читает SQLite через
собственное `mode=ro`-соединение. Проверки существования понятий кэшируются в
пределах документа. Соседи запрашиваются один раз на уникальное понятие документа,
не на каждое повторение слова. Индексы SQLite на `concepts(id)`, `relations(start)`
и `relations(end)` желательно проверить при подготовке базы; источник не меняется.

Разворачивание пар и запись Parquet выполняет DuckDB. Нет плотной N×N матрицы
или Python-списка всех пар. До разворачивания точно оценивается число рёбер по
частотам понятий. `max_edges_per_document` — предохранитель с ошибкой, **не обрезка**.
Экспорт PreparedGraph имеет отдельный `--max-edges`: его API хранит selected_edges
в памяти. Для больших документов можно оставить Parquet и экспортировать граф
понятий. Исходные тексты/слова хранятся в памяти; это не потоковый reader гигантского
корпуса. Время и пиковое потребление ресурсов на полном корпусе требуют измерения.

Начальная конфигурация: 4 процесса × 1 поток DuckDB, лимит DuckDB 1GB **на процесс**
(не лимит всей памяти Python). В Colab SQLite и промежуточные результаты держать
на `/content`, Google Drive использовать для копирования завершённых результатов.
GPU на этой стадии не требуется; далее применимы имеющиеся численные backend'ы.

Каждый документ публикуется атомарно после записи хешей. Повреждённый результат
при RESUME вызывает ошибку, а не тихий пересчёт поверх данных. При изменении текста,
БД, кода или смысловых настроек требуется новый output. Число процессов можно
изменить при восстановлении. SQLite должна быть замороженным checkpointed snapshot;
активная WAL-база отклоняется. Одновременно допускается один координатор на output.
После аварийного убийства координатора может остаться `.writer.lock`: удалять его
можно только после проверки, что прежний процесс действительно завершён.

## Проверки

```bash
python -m pip install -e '.[text,dev]'
python -m nltk.downloader stopwords
python -m pytest -q tests/test_text_corpus_words.py tests/test_text_corpus.py
```

Интеграционные тесты используют настоящую библиотеку ConceptNet Local и DuckDB,
но **маленькую искусственную SQLite-базу** с помеченными `fixture:*` утверждениями.
Проверяются все повторения, направления, несколько отношений, неизвестные слова,
пустые документы, отсутствие междокументных рёбер, контроль размера, RESUME,
перенос результатов, равенство последовательного/параллельного запуска и экспорт.
Это проверка программного контракта, не измерение покрытия полного ConceptNet.

Для коммита `6e30867ee8d17f47ede47363ab5a3911b7b692bc` обе проверки GitHub Actions
(Python 3.12 и 3.13) завершились успешно: 15 тестов в каждой среде.
Запуск: https://github.com/SemanticMap/semgraphex/actions/runs/43698163329
Полный пользовательский корпус и многогигабайтная база в этой проверке не запускались;
опциональная модель spaCy не загружалась.

Источники API: https://pypi.org/project/conceptnet-local/ и
https://github.com/lukDev/conceptnet-local/blob/main/conceptnet_local/_cn_service.py
DuckDB/Parquet: https://duckdb.org/docs/stable/data/parquet/overview
