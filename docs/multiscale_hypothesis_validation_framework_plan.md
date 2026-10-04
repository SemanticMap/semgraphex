# План разработки фреймворка проверки гипотезы о многоуровневой организации семантических сетей

## 0. Назначение документа

Этот документ задаёт пошаговый план развития `semgraphex` из набора отдельных исследовательских pipeline в единый воспроизводимый фреймворк проверки центральной гипотезы диссертационного исследования:

> **Большие семантические сети обладают статистически устойчивой многоуровневой организацией.**

Под «подтверждением» понимается не сам факт построения иерархии каким-либо алгоритмом, а согласие нескольких независимых измерительных каналов:

1. статистическая предпочтительность многоуровневой модели относительно плоских и нулевых моделей;
2. устойчивость уровней к повторной выборке и возмущениям;
3. независимая структурно-информационная подтверждаемость через повторяющиеся графовые формы и MDL;
4. согласованность graphon/graphex-представлений между уровнями и выборками;
5. сохранение или выявление макроскопических динамических мод;
6. внешняя семантическая интерпретируемость, не использованная при построении основной иерархии.

Фреймворк должен позволять **как подтвердить, так и опровергнуть** гипотезу.

---

# 1. Исходная база и принципы развития

Ветка создаётся от:

`feature/recursive-lossless-graph-dictionary-v2`

и должна повторно использовать уже реализованные контракты, а не дублировать их.

## 1.1. Уже существующие компоненты, которые необходимо сохранить

### Данные и воспроизводимость

- `src/semmap_haken/conceptnet.py` — потоковая подготовка ConceptNet с сохранением URI, направления и provenance;
- `src/semmap_haken/graph_build.py` — sparse CSR-представление;
- `src/semmap_haken/config.py`, `manifest.py` — конфигурация, lineage, checksums;
- Colab/Drive NEW/RESUME-контур.

### Рекурсивная грамматика и exact-кодирование

- `graph_dictionary.py`;
- `recursive_grammar.py`;
- `grammar_codec.py`, `grammar_binary.py`, `grammar_streams.py`;
- `hierarchy_codec.py`;
- `occurrence_index.py`;
- `compression_analysis.py`.

Критический инвариант сохраняется: exact graph representation и статистическая модель остаются разными сущностями.

### Эмпирические graphon/graphex-проекции

- `dictionary_graphex.py` — текущий grammar-induced typed block kernel;
- `multiscale_graph_model.py` — сопоставление соседних эмпирических проекций.

Эти модули следует обобщить до partition-agnostic слоя. Текущий grammar-induced projection остаётся одним частным источником разбиений.

### Динамика

- `operators.py`;
- `modes.py`;
- `dynamics.py`;
- `wishart_dynamics.py`.

Спектральные и динамические вычисления должны использоваться как независимый канал валидации, а не как источник основной иерархии.

### Wishart / WL / VF2 / GloVe

Сохраняются как вспомогательные измерительные и candidate-generation инструменты:

- Wishart — density-based validator;
- WL — дешёвый структурный фильтр;
- VF2 — exact structural identity;
- GloVe — внешняя семантическая проверка, а не вход основной nSBM-модели.

---

# 2. Архитектурный принцип

Главный архитектурный инвариант:

```
детектор иерархии != валидатор иерархии
```

Центральный поток:

```text
Prepared typed directed graph G0
        |
        v
Statistical detector
Directed typed degree-corrected Nested SBM
        |
        v
Hierarchy P0 < P1 < ... < PL
        |
        +-------------------+--------------------+
        |                   |                    |
        v                   v                    v
Graphon/Graphex        Grammar/MDL          Dynamics
projections            validation            validation
        |                   |                    |
        +-------------------+--------------------+
                            |
                            v
                  Stability / null models
                            |
                            v
                   Hypothesis evidence report
```

Ни один отдельный канал не должен сам по себе объявлять гипотезу подтверждённой.

---

# 3. Предлагаемая структура пакета

Чтобы не продолжать разрастание плоского `src/semmap_haken/`, новый код размещать в отдельном подпакете:

```text
src/semmap_haken/hypothesis/
    __init__.py
    contracts.py
    dataset.py
    hierarchy.py
    nsbm_backend.py
    null_models.py
    perturbations.py
    sampling.py
    partition_metrics.py
    block_projection.py
    graphex_inference.py
    graphex_sampling.py
    grammar_validation.py
    dynamic_validation.py
    semantic_validation.py
    evidence.py
    runner.py
```

CLI:

```text
src/semmap_haken/hypothesis_cli.py
```

entrypoint:

```toml
semmap-hypothesis = "semmap_haken.hypothesis_cli:main"
```

Notebook должен быть тонким и CLI-backed.

---

# 4. Базовые контракты данных

## 4.1. CanonicalSemanticGraph

Новый аналитический слой не должен читать ad hoc JSON/NPZ напрямую.

В `contracts.py` ввести immutable-контракт:

```python
CanonicalSemanticGraph(
    node_count,
    relation_layers,
    node_ids,
    original_mass,
    directed,
    weighted,
    provenance,
    source_checksum,
)
```

Требования:

- не терять направление;
- не смешивать relation layers;
- не интерпретировать ConceptNet weight как вероятность;
- сохранять соответствие level-0 узлам;
- не строить dense NxN матрицы.

## 4.2. HierarchyResult

```python
HierarchyResult(
    levels,
    assignments,
    block_edge_counts,
    block_masses,
    description_lengths,
    posterior_metadata,
    backend_metadata,
)
```

Для каждого уровня должны быть явно сохранены:

- membership исходных level-0 узлов;
- число блоков;
- block masses;
- relation-specific edge counts;
- MDL / entropy / likelihood;
- parent-child mapping;
- seed;
- backend/version.

## 4.3. StatisticalProjection

Обобщить идею `dictionary_graphex.py`:

```python
StatisticalProjection(
    level,
    masses,
    relation_block_intensities,
    degree_correction,
    W,
    S,
    I,
    residual_diagnostics,
)
```

`residual` не является компонентом graphex.

---

# 5. Этап 0. Зафиксировать научный контракт

## Цель

До написания нового алгоритмического кода превратить гипотезу в машинно-проверяемые критерии.

## Реализация

Добавить:

`src/semmap_haken/hypothesis/contracts.py`

и:

`docs/hypothesis_validation_contract.md`

Определить четыре группы критериев.

### H1. Статистическая многоуровневость

Сравнение:

- configuration model;
- flat SBM;
- flat DC-SBM;
- Nested DC-SBM.

Минимальный критерий:

```text
DL_nested < DL_flat
```

с обязательным отчётом абсолютной и относительной разницы.

### H2. Устойчивость

Для уровня `l` сохранять:

- AMI;
- NMI;
- VI;
- variation across seeds;
- variation across perturbations.

### H3. Независимая валидация

Для каждого уровня собирать:

- grammar MDL gain;
- motif/WL enrichment;
- dynamic agreement;
- semantic coherence.

### H4. Graphex consistency

Для p-samples и независимых подвыборок оценивать совместимость `W,S,I` после допустимой нормировки/согласования.

## Acceptance gate

Ни один дальнейший модуль не должен содержать hard-coded формулировку «hypothesis confirmed». Финальное решение формирует только `evidence.py`.

---

# 6. Этап 1. Отделить level-0 научные данные от старых coarsened artifacts

## Цель

Основная статистическая иерархия должна строиться из исходного подготовленного directed typed level-0 графа, а не из уже сжатого Wishart-графа.

## Работы

1. Создать `hypothesis/dataset.py`.
2. Реализовать загрузку prepared ConceptNet в единый canonical contract.
3. Проверять:
   - shape relation layers;
   - orientation;
   - node count;
   - selected edge provenance;
   - checksums;
   - отсутствие скрытой symmetrization.
4. Если источник уже был симметризован, помечать run:
   `directionality_recoverable=false`.

## Тесты

- directed reciprocal edges;
- parallel relation records;
- self-loops;
- empty layers;
- OOV/isolates;
- exact node-ID mapping;
- deterministic checksum.

## Acceptance gate

Любой nSBM-run обязан ссылаться на неизменяемый level-0 source checksum.

---

# 7. Этап 2. Добавить основной статистический детектор: Directed Typed DC-nSBM

## 7.1. Backend abstraction

Создать:

`hypothesis/nsbm_backend.py`

Интерфейс:

```python
class HierarchyBackend(Protocol):
    def fit(graph, config) -> HierarchyResult: ...
```

Первая реализация:

```text
GraphToolNestedSBMBackend
```

## 7.2. Почему backend изолируется

`graph-tool` не следует превращать в обязательную core pip dependency:

- пакет имеет сложную системную установку;
- Colab/CI могут использовать отдельное окружение;
- научный контракт не должен зависеть от конкретной библиотеки.

Backend обязан сохранять:

- graph-tool version;
- inference model flags;
- degree correction;
- layered/covariate setup;
- entropy/description length;
- seed;
- sweeps/equilibration metadata.

## 7.3. Typed relations

Основной вариант — shared node hierarchy + relation-specific edge layers/covariates.

Не агрегировать `IsA`, `RelatedTo`, `UsedFor` и другие отношения в один неразмеченный edge count без отдельной ablation.

## 7.4. Directed model

Направление используется в основной модели.

Symmetric projection допускается только как ablation.

## 7.5. Baseline fitting

Один CLI-run должен уметь получить:

```text
configuration
flat_sbm
flat_dc_sbm
nested_dc_sbm
```

в одном lineage.

## Artifacts

```text
hypothesis/nsbm/
    backend.json
    model_comparison.json
    hierarchy.json
    level_000_membership.npy
    ...
    level_L_membership.npy
    block_statistics/
```

## Acceptance gate

На smoke fixture:

- hierarchy round-trip membership deterministic under fixed seed;
- flat/nested DL comparable в одной системе измерения;
- relation counts conserved;
- исходный граф не мутируется.

---

# 8. Этап 3. Нулевые модели

Это обязательный этап до интерпретации найденной иерархии.

## Новый модуль

`hypothesis/null_models.py`

## Null-0

Erdos-Renyi / density-matched — только sanity baseline.

## Null-1

Degree-preserving configuration model.

## Null-2

Directed in/out-degree preserving model.

## Null-3 — основной

Relation-specific directed degree-preserving null:

для каждого типа отношения отдельно сохраняются in/out degree sequences настолько точно, насколько позволяет выбранный edge-swap/generative алгоритм.

## Требования

- deterministic seed streams;
- запрещены silent edge drops;
- отчёт о degree error;
- отчёт о relation count error;
- проверка mixing / number of swaps;
- возможные multi-edge/self-loop policies фиксируются явно.

## Эксперимент

Для каждого null ensemble:

```text
G_null_1 ... G_null_N
       |
       v
same nSBM pipeline
```

Сравнивать:

- hierarchy depth;
- DL gain nested vs flat;
- block count;
- stability;
- dynamic scales;
- grammar gain.

## Acceptance gate

Основной evidence report обязан показывать empirical null distribution, а не только одно случайное сравнение.

---

# 9. Этап 4. Perturbation и resampling stability

## Модули

- `perturbations.py`
- `sampling.py`
- `partition_metrics.py`

## Набор возмущений

### Edge dropout

```text
p = 0.95, 0.90, 0.80, 0.70
```

### Relation dropout

Отдельно измерять чувствительность к удалению конкретных relations.

### Node sampling

Стратифицированное и обычное.

### Bootstrap / subsampling

Несколько повторов каждого режима.

### Seed sensitivity

Повторять nSBM inference с разными seeds.

## Метрики

Для partitions:

- AMI;
- NMI;
- VI.

Для иерархий:

- matching уровней по минимуму VI;
- parent-child consistency;
- block mass drift;
- hierarchy depth drift.

## Важное правило

Не сравнивать «уровень 2» с «уровнем 2» только по номеру. Уровни должны сначала быть согласованы по масштабу/массе/partition distance.

## Acceptance gate

`stability_report.json` должен содержать confidence intervals по повторным запускам.

---

# 10. Этап 5. Обобщить block projection до nSBM-driven graphon family

## Рефакторинг

Текущий:

`dictionary_graphex.py`

разделить концептуально на:

1. generic block projection;
2. grammar adapter;
3. nSBM adapter.

Новый общий модуль:

`hypothesis/block_projection.py`

## W на уровне l

Для relation `r`:

```text
W_l^r(a,b) = observed relation intensity between nSBM blocks a,b
```

Сохранять одновременно:

- record intensity;
- weight intensity;
- block mass;
- exposure;
- credible/confidence intervals где доступны.

## Degree correction

Не терять индивидуальную degree heterogeneity при интерпретации блока.

Не утверждать, что DC-SBM автоматически является классическим piecewise-constant graphon.

## Inter-level projection

Для каждого уровня:

```text
W_0, W_1, ..., W_L
```

Сохранять aligned statistics по общей level-0 mass measure.

## Acceptance gate

Суммарные relation counts, массы и exposures должны быть проверяемо согласованы с level-0 источником.

---

# 11. Этап 6. Реальная graphex inference: W, S, I

## Новый модуль

`hypothesis/graphex_inference.py`

## 11.1. W

Источник: nSBM block intensity + degree correction.

В `W` входят и intra-block, и inter-block связи.

## 11.2. S

Не использовать правило:

```text
degree == 1 => S
```

как окончательную классификацию.

Наблюдаемые pendant vertices используются как evidence для star component, но происхождение должно оцениваться статистически.

Минимальный первый вариант:

- empirical block-conditional star rate;
- expected pendant rate under fitted W;
- excess pendant count = evidence for S.

Следующий вариант:

latent assignment / Bayesian mixture.

## 11.3. I

Аналогично:

- считать isolated pairs;
- оценивать expected isolated-pair count under fitted W;
- excess isolated-pair mass интерпретировать как evidence for dust I.

## 11.4. Residual

Существующий grammar residual переименовывать нельзя.

```text
Delta != I
```

Residual остаётся exact correction/codec artifact.

## Artifacts

```text
hypothesis/graphex/level_XXX/
    W.json
    S.json
    I.json
    fit_diagnostics.json
    residual_diagnostics.json
```

## Acceptance gate

Каждый компонент обязан содержать поле:

```text
estimation_semantics
```

с явным указанием, является ли он:

- fitted;
- empirical;
- excess-over-W;
- latent posterior;
- codec-only.

---

# 12. Этап 7. p-sampling consistency

## Новый модуль

`hypothesis/graphex_sampling.py`

## Протокол

Для:

```text
p = 1.0, 0.8, 0.5, 0.25, 0.125
```

1. независимо сохранять узлы с вероятностью p;
2. строить induced subgraph;
3. удалять isolated vertices согласно выбранной graphex sampling semantics;
4. независимо повторять nSBM + graphex estimation;
5. сравнивать rescaled projections.

## Метрики

Первая версия:

- block-mass TV;
- relation intensity weighted L1;
- degree distribution distance;
- component-size distance;
- star/dust rate deviations.

Продвинутая версия:

- sampling-based graph distance;
- cut-like diagnostics только если математически обоснованы для выбранного режима.

## Acceptance gate

Текущие `multiscale_graph_model.py` метрики не должны переименовываться в graphex distance без отдельного доказательства.

---

# 13. Этап 8. Независимая grammar/MDL validation

## Новый adapter

`hypothesis/grammar_validation.py`

Он не должен переписывать existing grammar codec.

Он агрегирует уже существующие outputs:

- shape frequencies;
- accepted occurrences;
- rule DAG;
- compact binary baseline;
- residual share;
- measured archive bytes.

## Главный вопрос

Совпадают ли области высокой nSBM-структурированности с областями структурной повторяемости?

Для каждого nSBM блока/уровня считать:

- exact-shape enrichment;
- WL-signature enrichment;
- grammar reuse rate;
- local MDL gain;
- residual fraction.

## Контроль

Сравнивать те же показатели на null graphs.

## Критерий

Не считать «частый motif» доказательством семантического primitive.

Сильное свидетельство:

```text
nSBM significance
AND perturbation stability
AND grammar MDL gain
AND null-model separation
```

---

# 14. Этап 9. Динамическая независимая валидация

## Новый модуль

`hypothesis/dynamic_validation.py`

Переиспользовать:

- normalized adjacency;
- existing slow eigenmodes;
- stationary mass;
- principal-angle machinery;
- Wishart dynamic metrics.

## Добавить Markov-time scale scan

Для набора времён `t` измерять устойчивые partitions/flow-retention scales.

Первая реализация может использовать:

- random-walk transition operator;
- spectral approximation;
- partition retention/autocovariance diagnostics.

Полноценный Markov Stability optimiser допустим как отдельный backend.

## Сравнение

```text
nSBM levels <-> dynamic plateaus
```

Нельзя просто сравнивать одинаковые номера уровней.

Использовать:

- VI/AMI между согласованными partitions;
- slow-space principal angles;
- eigenvalue retention;
- mixing-time changes.

## Acceptance gate

Dynamic validator никогда не получает nSBM labels во время собственной оптимизации.

---

# 15. Этап 10. Внешняя семантическая валидация

## Новый модуль

`hypothesis/semantic_validation.py`

Основная nSBM-иерархия строится без GloVe.

После fit embeddings используются только как внешняя информация.

## Метрики

По каждому блоку:

- mean pairwise cosine;
- centroid dispersion;
- nearest-block separation;
- relation-label entropy;
- lexical/category enrichment.

## Контроль

- random blocks той же массы;
- degree-matched random blocks;
- null graph hierarchy.

## Acceptance gate

Semantic coherence считается дополнительным evidence, а не условием существования статистической структуры.

---

# 16. Этап 11. Итоговый evidence engine

## Новый модуль

`hypothesis/evidence.py`

Он собирает все независимые результаты.

## Выходы

```text
hypothesis_report.json
hypothesis_report.md
evidence_matrix.csv
```

## Evidence matrix

Строки — уровни/масштабы.

Колонки:

- nSBM DL gain;
- null-model z/quantile;
- bootstrap stability;
- perturbation stability;
- p-sampling consistency;
- grammar MDL gain;
- grammar enrichment;
- dynamic agreement;
- semantic coherence;
- W/S/I fit diagnostics.

## Итоговый статус

Не бинарный «истина/ложь», а:

```text
supported
partially_supported
not_supported
inconclusive
```

Каждый статус должен ссылаться на явно определённые gates из scientific contract.

---

# 17. CLI и orchestration

## Новый CLI

```bash
semmap-hypothesis prepare ...
semmap-hypothesis fit-hierarchy ...
semmap-hypothesis nulls ...
semmap-hypothesis stability ...
semmap-hypothesis project ...
semmap-hypothesis graphex ...
semmap-hypothesis grammar ...
semmap-hypothesis dynamics ...
semmap-hypothesis semantics ...
semmap-hypothesis report ...
semmap-hypothesis run-all ...
```

Каждая стадия:

- resumable;
- content-addressed/fingerprinted;
- atomically publishes artifacts;
- не перезаписывает совместимые completed stages;
- invalidates downstream stage при изменении upstream checksum/config.

---

# 18. Конфигурация

Добавить отдельные конфиги:

```text
configs/hypothesis/
    smoke.yaml
    conceptnet_10k.yaml
    conceptnet_100k.yaml
    conceptnet_100k_l4.yaml
```

Структура:

```yaml
hypothesis:
  detector:
    backend: graph_tool_nsbm
    directed: true
    degree_corrected: true
    relations: layered

  null_models:
    replicates: 20

  perturbations:
    edge_keep: [0.95, 0.90, 0.80, 0.70]
    replicates: 10

  p_sampling:
    probabilities: [1.0, 0.8, 0.5, 0.25, 0.125]
    replicates: 10

  validation:
    grammar: true
    dynamics: true
    semantics: true
```

Все экспериментальные thresholds должны находиться в config/report, а не быть скрыты внутри кода.

---

# 19. Artifact layout

```text
runs/<RUN_ID>/
    input.json
    manifest.json

    hypothesis/
      contract.json

      detector/
        model_comparison.json
        hierarchy.json
        levels/

      null_models/
        null_000/
        ...
        summary.json

      perturbations/
        ...
        stability_report.json

      sampling/
        ...
        p_sampling_report.json

      projections/
        level_000/
        ...
        multiscale_report.json

      graphex/
        level_000/
        ...

      grammar_validation/
        report.json

      dynamic_validation/
        report.json

      semantic_validation/
        report.json

      evidence/
        evidence_matrix.csv
        hypothesis_report.json
        hypothesis_report.md

    COMPLETED
```

`COMPLETED` записывается только после проверки scientific evidence report и artifact checksums.

---

# 20. Colab/L4 стратегия

Сохранить существующий принцип:

```text
Google Drive = durable storage
/content     = compute workspace
```

## CPU/GPU

nSBM graph-tool, вероятно, будет CPU-bound.

GPU оставлять для:

- embeddings;
- kNN;
- FGW diagnostics;
- возможно sparse tensor experiments.

Не переносить задачу на GPU только ради факта использования L4.

## Parallelism

Параллелить независимые:

- null replicas;
- perturbation replicas;
- p-samples;
- semantic metrics.

Но не допускать nested oversubscription:

```text
outer experiment workers x inner graph-tool threads
```

должны делить единый CPU budget.

## Notebook

Добавить:

`notebooks/18_multiscale_hypothesis_validation_cn100k_colab.ipynb`

Notebook:

- параметры;
- Drive mount;
- environment preflight;
- CLI invocation;
- artifact summary;
- plots.

Вся тяжёлая логика остаётся в package.

---

# 21. Testing strategy

Фреймворк научный, поэтому обычных unit tests недостаточно.

## 21.1. Unit tests — около 60%

Фокус:

- contracts;
- partition metrics;
- block exposures;
- null-model invariants;
- p-sampling;
- W/S/I accounting;
- evidence gates.

## 21.2. Integration tests — около 30%

Tiny typed directed graphs с заранее известной структурой:

1. flat SBM;
2. two-level hierarchical SBM;
3. degree-heterogeneous DC-SBM;
4. no-community configuration graph;
5. explicit star-heavy graph;
6. dust-heavy graph;
7. mixed W/S/I synthetic graph.

Проверять не только «код не упал», а восстановление ожидаемого свойства.

## 21.3. E2E/scientific acceptance — около 10%

Smoke pipeline:

```text
prepare
-> nSBM
-> nulls
-> perturbation
-> projection
-> graphex
-> grammar
-> dynamics
-> evidence report
```

## Обязательные regression tests

Существующие grammar exact roundtrip тесты не должны измениться.

Новый framework не должен менять exact archive semantics.

---

# 22. Synthetic truth benchmarks

До ConceptNet обязательно создать генераторы с известной истинной моделью.

## Benchmark A

Flat DC-SBM.

Ожидание: nested hierarchy не должна давать убедительное искусственное преимущество.

## Benchmark B

Two-level hierarchical DC-SBM.

Ожидание: detector восстанавливает два естественных масштаба.

## Benchmark C

Typed hierarchical SBM.

Ожидание: typed model выигрывает у relation-collapsed ablation.

## Benchmark D

Synthetic graphex W+S+I.

Ожидание: оценка различает excess stars/dust относительно fitted W.

## Benchmark E

Repeated grammar motifs without block structure.

Нужен для проверки, что grammar repeatability и nSBM hierarchy действительно независимы.

## Benchmark F

Strong block structure without repeated exact motifs.

Нужен обратный контроль.

Эти два последних benchmark особенно важны против кругового доказательства.

---

# 23. Scientific ablations

Обязательные сравнения:

1. directed vs symmetrized;
2. typed vs relation-collapsed;
3. degree-corrected vs ordinary SBM;
4. nested vs flat;
5. W-only vs W+S+I;
6. nSBM hierarchy vs Wishart hierarchy;
7. topology-only vs GloVe-assisted candidate discovery;
8. original graph vs nulls;
9. full graph vs p-samples;
10. exact grammar residual included vs ignored.

---

# 24. Performance plan

Не оптимизировать до корректности statistical contract.

После smoke correctness:

## Profile

Для каждой фазы сохранять:

- wall time;
- CPU time;
- peak RSS;
- GPU memory;
- input/output bytes;
- worker count;
- backend version.

## Масштабы

Последовательно:

```text
1k -> 10k -> 100k -> 1M where feasible
```

Нельзя переходить к 100k до прохождения synthetic truth benchmarks.

---

# 25. Порядок реализации по PR/commit-сериям

## Phase A — scientific foundation

1. contracts;
2. canonical dataset adapter;
3. partition metrics;
4. synthetic hierarchy generators.

**Gate:** unit + synthetic truth.

## Phase B — nSBM detector

5. backend abstraction;
6. graph-tool adapter;
7. flat/nested/DC model comparison;
8. hierarchy artifacts.

**Gate:** known hierarchical SBM recovered.

## Phase C — significance

9. null models;
10. perturbations;
11. bootstrap/resampling.

**Gate:** true hierarchy separates from nulls.

## Phase D — graph limits

12. generic block projection;
13. nSBM W-family;
14. W/S/I inference;
15. p-sampling consistency.

**Gate:** synthetic graphex test.

## Phase E — independent validators

16. grammar adapter;
17. dynamic validator;
18. semantic validator.

**Gate:** validators can disagree without breaking pipeline.

## Phase F — evidence engine

19. unified evidence matrix;
20. falsification statuses;
21. markdown report.

**Gate:** report contains both positive and negative evidence.

## Phase G — ConceptNet scale

22. 10k pilot;
23. 100k CPU/Colab run;
24. bottleneck remediation;
25. second independent semantic graph.

**Gate:** reproducible cross-dataset report.

---

# 26. Что не делать в первой версии

Не включать в MVP:

- FI-KAN;
- multifractal fitting;
- quantum algorithms;
- full nonlinear Haken center-manifold analysis;
- neural graphon as primary estimator;
- FGW all-pairs на 100k;
- попытку доказать теорему о сходимости graphex только экспериментом;
- автоматическое объявление recurring motifs «семантическими примитивами».

Это отдельные последующие исследования.

---

# 27. Главные риски и меры

## Риск 1. nSBM строит иерархию даже там, где научно значимого уровня нет

Мера:

- flat baseline;
- null ensembles;
- posterior/MDL;
- synthetic negative controls.

## Риск 2. Degree heterogeneity маскируется под communities

Мера:

- DC-SBM основной;
- ordinary SBM только ablation.

## Риск 3. Relation collapse создаёт ложные блоки

Мера:

- typed/layered основной режим;
- collapse только ablation.

## Риск 4. Graphex W/S/I неидентифицируемы из одного finite graph

Мера:

- не делать deterministic edge partition научным выводом;
- excess-over-W diagnostics;
- p-sampling;
- posterior uncertainty.

## Риск 5. Grammar validation становится круговым

Мера:

- nSBM не получает grammar features;
- grammar не получает nSBM labels при discovery;
- сопоставление только после независимого fit.

## Риск 6. Dynamic validation становится зависимой

Мера:

- Markov/dynamic partitioning выполняется без nSBM labels;
- matching только post hoc.

## Риск 7. Colab runtime слишком мал для nSBM replicas

Мера:

- resumable replicas;
- independent run shards;
- Drive checkpoints;
- ограничение worker budget.

---

# 28. Definition of Done для framework v1

Framework v1 считается готовым, когда выполнены все условия:

1. Один CLI запускает полный reproducible pipeline.
2. Directed typed level-0 graph имеет immutable checksum.
3. Flat и Nested DC-SBM сравниваются одной метрикой model evidence/description length.
4. Есть relation-specific degree-preserving null ensemble.
5. Есть perturbation stability report.
6. Есть p-sampling experiment.
7. Для каждого уровня есть W projection.
8. S и I не подменяются codec residual.
9. Grammar validation использует существующий exact codec и compact binary baseline.
10. Dynamics выполняется независимо.
11. Semantic embeddings используются только post hoc.
12. Synthetic positive и negative controls проходят.
13. Финальный report умеет выдавать `not_supported` и `inconclusive`.
14. NEW/RESUME идентичны по scientific artifacts.
15. ConceptNet-100k run не объявляется завершённым без полного evidence report и `COMPLETED`.
16. Второй независимый семантический граф проходит тот же protocol без изменения кода framework.

---

# 29. Первый практический инкремент

Рекомендуемый первый кодовый PR после этого планового документа:

**MVP-1: Statistical hierarchy foundation**

Состав:

- `hypothesis/contracts.py`;
- `hypothesis/dataset.py`;
- `hypothesis/partition_metrics.py`;
- `hypothesis/nsbm_backend.py`;
- synthetic flat/hierarchical DC-SBM fixtures;
- smoke config;
- CLI `fit-hierarchy`;
- tests.

Не включать пока graphex, grammar и dynamics.

Цель MVP-1:

```text
input graph
-> flat/DC/nested comparison
-> hierarchy artifacts
-> reproducible MDL evidence
```

После его прохождения добавлять null models. Это минимизирует риск построить сложный framework вокруг некорректного основного detector.

---

# 30. Целевая научная конструкция

После завершения всех этапов framework должен формировать объект:

```text
H(G) =
{
    statistical_hierarchy,
    null_evidence,
    stability,
    graphon_graphex_family,
    grammar_evidence,
    dynamic_evidence,
    semantic_evidence
}
```

и отвечать не на вопрос:

> «Удалось ли построить несколько уровней графа?»

а на вопрос:

> **«Есть ли в наблюдаемой семантической сети многоуровневая организация, которая статистически предпочтительна, устойчива к способу выборки, независимо проявляется в структурной повторяемости и динамике и допускает согласованное graphon/graphex-представление?»**

Именно этот вопрос должен быть главным контрактом программного фреймворка.
