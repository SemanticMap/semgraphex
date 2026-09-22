\# Agents.md — Haken-Coarsening Graphons for ConceptNet

\#\# 0\. Назначение

Этот файл — постановка исследовательской и программной задачи для ИИ-агентов проекта \*\*Haken-Coarsening Graphons for ConceptNet\*\*.

Цель: проверить, можно ли последовательно укрупнять семантический граф ConceptNet

\\\[  
G\_0 \\rightarrow G\_1 \\rightarrow G\_2 \\rightarrow \\cdots \\rightarrow G\_S  
\\\]

за счёт объединения вершин, сходных по участию в медленных/критических коллективных модах, при этом сохранять макродинамику и обнаруживать устойчивые крупномасштабные уровни семантической организации.

\*\*Важно:\*\* \\\`Haken-coarsening\\\` и \\\`Haken graphon estimator\\\` — рабочие названия предлагаемой исследовательской конструкции, а не канонические методы из литературы. Все сильные выводы должны подтверждаться экспериментом и baselines.

\---

\#\# 1\. Центральный исследовательский вопрос

\> Существуют ли в ConceptNet естественные масштабы coarse-graining, на которых число микроскопических вершин резко уменьшается, но маломерное пространство медленных коллективных мод и соответствующая макродинамика остаются устойчивыми?

Ищем интервалы масштаба \\(s=a,\\ldots,b\\), для которых

\\\[  
|V\_{s+1}| \\ll |V\_s|,  
\\\]

но

\\\[  
\\operatorname{span}\\{\\phi\_1^{(s)},\\ldots,\\phi\_r^{(s)}\\}  
\\approx  
\\operatorname{span}\\{\\phi\_1^{(s+1)},\\ldots,\\phi\_r^{(s+1)}\\}.  
\\\]

Такие интервалы далее называются \*\*semantic scale plateaus\*\*.

\---

\#\# 2\. Гипотезы

\#\#\# H1 — низкоразмерная макродинамика

Для корректно построенного графа ConceptNet \\(G\_0\\) существует динамическая модель

\\\[  
\\dot{\\mathbf x}=F(\\mathbf x,A),  
\\\]

в которой небольшое число медленных/критических мод

\\\[  
a\_1,\\ldots,a\_r,\\qquad r\\ll |V|  
\\\]

описывает основную долгоживущую динамику.

\#\#\# H2 — Haken-equivalence

Если две вершины имеют близкие координаты в пространстве параметров порядка

\\\[  
z\_i=(\\phi\_1(i),\\ldots,\\phi\_r(i)),  
\\\]

то их можно объединить с малой потерей макродинамической информации:

\\\[  
i\\sim\_H j \\iff d\_H(i,j)\<\\varepsilon.  
\\\]

\#\#\# H3 — slaving

В нелинейной версии быстрые моды могут быть приближённо выражены через медленные:

\\\[  
\\mathbf b(t)\\approx h(\\mathbf a(t)).  
\\\]

\#\#\# H4 — устойчивые масштабы

Последовательность

\\\[  
G\_0\\to G\_1\\to\\cdots  
\\\]

содержит интервалы, на которых стабильны число параметров порядка \\(r\_s\\), slow eigenspace, спектр медленных мод, выбранные dynamical observables и крупные coarse blocks.

\#\#\# H5 — нетривиальность

Результат не должен полностью объясняться degree distribution, hubs, standard community detection, ordinary spectral clustering или простой low-rank approximation.

\#\#\# H6 — семантическая интерпретируемость

После слепого topology/dynamics-only coarsening устойчивые coarse blocks должны хотя бы частично иметь осмысленную интерпретацию по ConceptNet labels и relation types.

\---

\#\# 3\. Теоретический статус

\#\#\# Установленная база

Разрешено опираться на:

\- Haken-style order parameters и slaving principle;  
\- slow/fast mode separation;  
\- finite-dimensional reduction of dissipative systems;  
\- spectral dynamics on graphs;  
\- quotient graphs и contraction hierarchies;  
\- graphons, step graphons и kernel operators;  
\- sparse-network/graphex caveats.

\#\#\# Исследовательский синтез

Не выдавать как установленный факт:

\- существование Haken-параметров порядка у ConceptNet;  
\- корректность объединения вершин только по Haken-модам;  
\- существование стандартного Haken graphon estimator;  
\- наличие semantic scale plateaus;  
\- параметризацию  
  \\\[  
  W\_s(x,y)\\approx W(x,y;a\_1^{(s)},\\ldots,a\_r^{(s)}).  
  \\\]

\---

\#\# 4\. Данные: ConceptNet 5.7

Использовать официальный dump assertions.

Официальная документация:

\- Downloads: https://github.com/commonsense/conceptnet5/wiki/Downloads  
\- Edge schema: https://github.com/commonsense/conceptnet5/wiki/Edges  
\- Relations: https://github.com/commonsense/conceptnet5/wiki/Relations  
\- URI hierarchy: https://github.com/commonsense/conceptnet5/wiki/URI-hierarchy  
\- Data license: CC BY-SA 4.0

Строка assertions содержит:

1\. assertion URI;  
2\. relation URI;  
3\. start concept URI;  
4\. end concept URI;  
5\. JSON metadata.

Метаданные включают \\\`weight\\\`, \\\`dataset\\\`, \\\`sources\\\`, \\\`license\\\`.

\*\*Критическое ограничение:\*\* ConceptNet \\\`weight\\\` — эвристический вес уверенности/информативности, а не эмпирическая вероятность перехода. Не интерпретировать его автоматически как Markov probability.

\---

\#\# 5\. Этапы данных

\#\#\# Stage A — small English

Цель:

\- 5k–20k узлов;  
\- English only;  
\- whitelist relations;  
\- threshold по weight;  
\- largest connected component.

\#\#\# Stage B — medium English

Цель:

\- 50k–200k узлов;  
\- только sparse representation.

\#\#\# Stage C — relation-specific

Построить отдельные графы минимум для:

\- \\\`RelatedTo\\\`;  
\- \\\`IsA\\\`;  
\- \\\`PartOf\\\`;  
\- \\\`HasA\\\`;  
\- \\\`Synonym\\\`;  
\- \\\`Antonym\\\`;  
\- \\\`Causes\\\`;  
\- \\\`UsedFor\\\`;  
\- \\\`HasProperty\\\`.

\#\#\# Stage D — multiplex

Исследовать relation-aware representation:

\\\[  
A=\\sum\_r \\alpha\_r A^{(r)}  
\\\]

и/или multiplex/supra-adjacency. Не начинать с этого до стабилизации однослойного baseline.

\---

\#\# 6\. Обязательная поправка на разреженность

ConceptNet — sparse, heterogeneous, hub-rich knowledge graph. Классический graphon наиболее естественен для плотного режима.

Вести две ветки.

\#\#\# Branch G — graphon-compatible

Использовать filtered/induced subgraphs, step graphon, block graphon, контролируемую нормировку и при необходимости sparse graphon estimators.

\#\#\# Branch X — sparse control

Работать напрямую с sparse adjacency, normalized adjacency, Laplacian, random-walk operators и optional graphex-inspired summaries.

Если plateau появляется только после искусственного уплотнения, трактовать его как возможный graphon artifact.

\---

\#\# 7\. Масштаб и время

Использовать:

\- \\(s\\) — coarsening scale;  
\- \\(t\\) — dynamical time.

На уровне \\(s\\):

\\\[  
G\_s=(V\_s,E\_s,A\_s).  
\\\]

Разбиение:

\\\[  
\\Pi\_s=\\{C\_1^{(s)},\\ldots,C\_{k\_s}^{(s)}\\}.  
\\\]

Переход:

\\\[  
G\_{s+1}=G\_s/\\Pi\_s.  
\\\]

Каждый supernode хранит исходные vertex IDs, mass, relation histogram, aggregate weights, provenance, parent/child mapping. Membership hierarchy должна быть обратимой.

\---

\#\# 8\. Построение графа

\#\#\# Nodes

По умолчанию вершина \= полный ConceptNet concept URI, например \\\`/c/en/dog\\\`.

Не склеивать автоматически леммы, senses, \\\`FormOf\\\` и синонимы. Каждая такая операция — отдельная ablation.

\#\#\# Edge mode U — symmetrized

Для первого baseline:

\\\[  
A\_{ij}=\\sum\_{e:i\\leftrightarrow j}w\_e.  
\\\]

\#\#\# Edge mode D — directed

Сохранять start/end после стабилизации undirected baseline.

\#\#\# Edge mode R — relation-aware

\\\[  
A\_{ij}=\\sum\_r\\alpha\_r A\_{ij}^{(r)}.  
\\\]

Все \\(\\alpha\_r\\) конфигурируемы.

\#\#\# Weight transforms

Поддержать:

\- binary;  
\- raw;  
\- \\\`log1p\\\`;  
\- capped;  
\- relation-normalized.

\---

\#\# 9\. Динамические модели

\#\#\# H0 — линейная диссипативная модель

Обязательный baseline:

\\\[  
\\dot{\\mathbf x}  
\=  
\-\\alpha\\mathbf x+\\beta S\\mathbf x,  
\\\]

где

\\\[  
S=D^{-1/2}AD^{-1/2}.  
\\\]

Якобиан:

\\\[  
J=-\\alpha I+\\beta S.  
\\\]

Если

\\\[  
S\\phi\_k=\\lambda\_k\\phi\_k,  
\\\]

то

\\\[  
\\gamma\_k=-\\alpha+\\beta\\lambda\_k.  
\\\]

Slow/critical candidates:

\\\[  
|\\operatorname{Re}\\gamma\_k|\\approx0.  
\\\]

\#\#\# H1 — слабонелинейная модель

Обязательная nonlinear branch:

\\\[  
\\dot{\\mathbf x}  
\=  
\-\\alpha\\mathbf x+\\beta S\\mathbf x-g\\,\\mathbf x^{\\circ3}.  
\\\]

Дополнительные nonlinearities разрешены, но cubic baseline сохранять.

\---

\#\# 10\. Кандидаты в параметры порядка

На каждом масштабе вычислять modes для одного или нескольких операторов:

1\. normalized adjacency;  
2\. random-walk operator;  
3\. Laplacian;  
4\. Jacobian \\(J\_s\\).

Для каждого mode хранить:

\- eigenvalue;  
\- growth/decay rate;  
\- relaxation time;  
\- participation ratio;  
\- localization score;  
\- bootstrap stability;  
\- cross-scale stability.

\*\*Запрещено:\*\* называть параметром порядка просто top eigenvector.

Кандидат должен демонстрировать комбинацию slow/critical timescale, spectral separation, reproducibility, scale persistence и nonlinear relevance.

\---

\#\# 11\. Выбор размерности \\(r\_s\\)

Не использовать один heuristic.

Реализовать:

\- eigengap;  
\- timescale gap;  
\- elbow/profile likelihood;  
\- perturbation stability;  
\- bootstrap consensus.

Выход:

\- spectrum plot;  
\- relaxation-time plot;  
\- eigengap plot;  
\- выбранный \\(r\_s\\);  
\- uncertainty.

\---

\#\# 12\. Haken embedding

Для узла \\(i\\):

\\\[  
z\_i^{(s)}  
\=  
(\\phi\_1^{(s)}(i),\\ldots,\\phi\_{r\_s}^{(s)}(i)).  
\\\]

Опционально:

\\\[  
z\_{ik}^{(s)}=w\_k\\phi\_k^{(s)}(i),  
\\\]

где \\(w\_k\\) учитывает eigenvalue, relaxation time или modal energy.

\---

\#\# 13\. Haken distance

Baseline:

\\\[  
d\_H(i,j)=\\|z\_i-z\_j\\|\_2.  
\\\]

Dynamics-aware вариант:

\\\[  
d\_H^{dyn}(i,j)  
\=  
\\frac1R  
\\sum\_{r=1}^{R}  
\\int\_0^T  
|x\_i^{(r)}(t)-x\_j^{(r)}(t)|^2dt.  
\\\]

Composite:

\\\[  
d\_H^\*=\\eta d\_H+(1-\\eta)d\_H^{dyn}.  
\\\]

\---

\#\# 14\. Итеративный Haken-coarsening

На каждом \\(s\\):

1\. построить operator;  
2\. оценить slow/critical modes;  
3\. выбрать \\(r\_s\\);  
4\. вычислить Haken embedding;  
5\. сформировать clusters/merge pairs;  
6\. построить quotient graph \\(G\_{s+1}\\);  
7\. заново вычислить dynamics;  
8\. измерить distortion;  
9\. проверить plateau/stop condition.

Поддержать merge strategies:

\- agglomerative;  
\- k-means;  
\- radius clustering;  
\- nearest-neighbor matching;  
\- connectivity-constrained matching.

Не фиксировать один способ как «Haken method».

\---

\#\# 15\. Quotient graph

Для \\(C\_a,C\_b\\).

\#\#\# Sum

\\\[  
A'\_{ab}=\\sum\_{i\\in C\_a,j\\in C\_b}A\_{ij}.  
\\\]

\#\#\# Mean density

\\\[  
A'\_{ab}  
\=  
\\frac{\\sum\_{i\\in C\_a,j\\in C\_b}A\_{ij}}  
{|C\_a||C\_b|}.  
\\\]

\#\#\# Mass-aware block form

\\\[  
m\_a=\\frac{|C\_a|}{|V\_s|}.  
\\\]

Хранить block intensity \\(B\_{ab}\\) и mass \\(m\_a\\).

\---

\#\# 16\. Graphon на каждом масштабе

Строить block/step representation:

\\\[  
W\_s(x,y)=B\_{ab},  
\\qquad  
x\\in I\_a,\\;y\\in I\_b.  
\\\]

Для больших графов не материализовать dense \\(N\\times N\\).

Хранить block masses, block intensities, ordering/alignment metadata и normalization metadata.

\---

\#\# 17\. Haken graphon hypothesis

Проверяем:

\\\[  
W\_s(x,y)\\approx W(x,y;a\_1^{(s)},\\ldots,a\_r^{(s)}).  
\\\]

Возможная nonlinear expansion:

\\\[  
W(x,y;\\mathbf a)  
\=  
W\_0(x,y)  
\+  
\\sum\_i a\_i\\Phi\_i(x,y)  
\+  
\\sum\_{ij}a\_i a\_j\\Phi\_{ij}(x,y)  
\+\\cdots.  
\\\]

Сравнить predictive ability против:

\- rank-\\(r\\) spectral approximation;  
\- SBM;  
\- smooth/block graphon;  
\- simple embedding decoder.

\---

\#\# 18\. Проверка slaving principle

В nonlinear model разделить:

\\\[  
\\mathbf a=(a\_1,\\ldots,a\_r)  
\\\]

и

\\\[  
\\mathbf b=(b\_{r+1},\\ldots).  
\\\]

Проверить:

\\\[  
\\mathbf b(t)\\approx h(\\mathbf a(t)).  
\\\]

Модели \\(h\\):

\- ridge;  
\- polynomial;  
\- sparse polynomial / SINDy-like;  
\- small MLP как upper-bound.

Метрики:

\- \\(R^2\\);  
\- NRMSE;  
\- held-out trajectory error;  
\- robustness.

Сильное свидетельство slaving требует out-of-sample generalization.

\---

\#\# 19\. Возмущения

Нельзя обосновывать Haken только статическим spectrum.

Запускать:

1\. single-node impulse;  
2\. activation локального semantic neighborhood;  
3\. random sparse activation;  
4\. Gaussian initial state;  
5\. activation relation-defined group;  
6\. hub-targeted activation.

Сохранять initial state, trajectories, modal amplitudes, relaxation times, coarse trajectories.

\---

\#\# 20\. Semantic scale plateau

Для каждого перехода:

\#\#\# Compression

\\\[  
C\_s=\\frac{|V\_s|}{|V\_{s+1}|}.  
\\\]

\#\#\# Dynamic distortion

Измерять:

\- slow-subspace distance;  
\- slow eigenvalue error;  
\- trajectory reconstruction error.

\#\#\# Structural distortion

Измерять:

\- degree-distribution change;  
\- edge-weight distribution change;  
\- motif statistics;  
\- block-graphon discrepancy;  
\- optional aligned cut-distance approximation.

\#\#\# Partition stability

Использовать:

\- ARI;  
\- NMI;  
\- variation of information;  
\- parent-child overlap.

Plateau \= минимум \\(L\\) последовательных уровней, где продолжается compression, \\(r\_s\\) стабилен, slow subspace стабилен и dynamic distortion мал.

Default \\(L=3\\), параметр конфигурируемый.

\---

\#\# 21\. Stop criterion

Не останавливать алгоритм по заранее заданному \\(k\\).

Ввести:

\\\[  
R\_s=  
\\frac{D\_s^{dyn}}  
{\\log(|V\_s|/|V\_{s+1}|)}.  
\\\]

Остановка при:

1\. резком росте \\(R\_s\\);  
2\. потере slow-subspace stability;  
3\. скачке \\(r\_s\\);  
4\. превышении reconstruction threshold;  
5\. слишком малом графе для надёжного spectral estimate.

Всегда сохранять полную траекторию coarsening.

\---

\#\# 22\. Baselines

Обязательные:

1\. random matching;  
2\. heavy-edge matching;  
3\. Leiden contraction;  
4\. spectral clustering/coarsening;  
5\. SBM/blockmodel;  
6\. WL/equitable partition — где применимо.

Опционально:

\- Kron/Schur;  
\- diffusion maps;  
\- PF/Koopman coarsening.

Сравнивать при одинаковом compression ratio.

Главный вопрос:

\> Какой метод при одинаковом сжатии лучше сохраняет выбранную коллективную динамику?

\---

\#\# 23\. Null models

\#\#\# N1 — degree-preserving rewiring

Если Haken modes сохраняются после разрушения семантической структуры при сохранении degree, вероятно метод выделяет степени/hubs.

\#\#\# N2 — weight shuffle

Перемешать веса по существующим рёбрам.

\#\#\# N3 — relation shuffle

Перемешать relation labels при сохранении topology.

\#\#\# N4 — node-label permutation

Для проверки interpretation. Topology-only coarsening не должен зависеть от текстовых labels.

\---

\#\# 24\. Семантическая интерпретация

Текстовые labels не использовать при построении topology-only partition.

После получения clusters для каждого supernode вывести:

\- top concepts;  
\- internal centrality;  
\- relation histogram;  
\- enriched terms;  
\- representative nodes;  
\- optional LLM-generated short description.

LLM labels — только presentation, не evidence.

\---

\#\# 25\. Стандартная таблица метрик

Каждый run сохраняет:

\- dataset version;  
\- checksum;  
\- language;  
\- relation filter;  
\- node count;  
\- edge count;  
\- directed/undirected;  
\- weighted/unweighted;  
\- scale \\(s\\);  
\- coarse node count;  
\- compression ratio;  
\- \\(r\_s\\);  
\- eigengap;  
\- slow-subspace distance;  
\- slow eigenvalue error;  
\- trajectory error;  
\- partition stability;  
\- runtime;  
\- peak memory.

Опционально: graphon distance, motif preservation, semantic coherence, bootstrap intervals.

\---

\#\# 26\. Reproducibility

Каждый run воспроизводится из:

\- YAML config;  
\- dataset checksum;  
\- git commit;  
\- random seed;  
\- environment lock.

Publication-grade результаты должны запускаться через CLI, а не только notebook.

Пример:

\\\`\\\`\\\`bash  
python \-m semmap\_haken.prepare \--config configs/conceptnet\_en\_small.yaml  
python \-m semmap\_haken.run \--config configs/haken\_linear\_small.yaml  
python \-m semmap\_haken.evaluate \--run runs/\<run\_id\>  
\\\`\\\`\\\`

\---

\#\# 27\. Рекомендуемая структура репозитория

\\\`\\\`\\\`text  
.  
├── AGENTS.md  
├── README.md  
├── pyproject.toml  
├── configs/  
│   ├── conceptnet\_en\_small.yaml  
│   ├── conceptnet\_en\_medium.yaml  
│   ├── haken\_linear.yaml  
│   ├── haken\_nonlinear.yaml  
│   └── baselines.yaml  
├── data/  
│   ├── raw/  
│   ├── interim/  
│   └── processed/  
├── src/  
│   └── semmap\_haken/  
│       ├── conceptnet.py  
│       ├── graph\_build.py  
│       ├── operators.py  
│       ├── dynamics.py  
│       ├── modes.py  
│       ├── haken\_embedding.py  
│       ├── coarsen.py  
│       ├── quotient.py  
│       ├── graphon.py  
│       ├── plateau.py  
│       ├── metrics.py  
│       ├── baselines.py  
│       ├── null\_models.py  
│       ├── semantic\_labels.py  
│       └── cli.py  
├── tests/  
├── runs/  
└── reports/  
\\\`\\\`\\\`

\---

\#\# 28\. Роли ИИ-агентов

\#\#\# Agent A — Data Engineer

Скачать и проверить ConceptNet, парсить TSV+JSON, построить deterministic node map и sparse graph, подготовить data report.

\#\#\# Agent B — Spectral/Dynamics Researcher

Реализовать operators, linear/nonlinear dynamics, spectrum, relaxation times, perturbations и slow-mode selection.

\#\#\# Agent C — Coarsening Engineer

Реализовать Haken embedding, merge algorithms, quotient graph, contraction log и hierarchy.

\#\#\# Agent D — Graphon Researcher

Реализовать step/block graphons, alignment, cross-scale comparison, low-rank approximation и тест \\(W(x,y;\\mathbf a)\\).

\#\#\# Agent E — Evaluation/Falsification

Реализовать baselines, null models, plateau detection, bootstrap, statistics. Активно пытаться опровергнуть H1–H6.

\#\#\# Agent F — Scientific Auditor

Проверять source-vs-hypothesis, отсутствие semantic leakage, reproducibility, graphon applicability, корректность термина order parameter и сравнения eigenspaces. Может блокировать сильные выводы.

\---

\#\# 29\. Milestones

\#\#\# M0 — Data smoke test

\- installation works;  
\- sample parses;  
\- 1k–5k node graph built;  
\- sparse adjacency round-trip passes.

\#\#\# M1 — Linear Haken baseline

\- normalized operator;  
\- reproducible top modes;  
\- trajectory solver validated;  
\- \\(r\\) diagnostics generated.

\#\#\# M2 — One-step coarsening

\- Haken embedding;  
\- clusters;  
\- quotient;  
\- contraction mapping;  
\- dynamics on coarse graph.

\#\#\# M3 — Multi-scale hierarchy

\- \>=5 levels;  
\- metrics per level;  
\- parent-child hierarchy;  
\- no fixed final \\(k\\).

\#\#\# M4 — Plateau detector

\- planted synthetic hierarchy gives intended plateau;  
\- null/random graph does not produce equivalent plateau.

\#\#\# M5 — ConceptNet small

\- Haken hierarchy;  
\- baselines;  
\- null tests;  
\- post-hoc semantic interpretation;  
\- complete report.

\#\#\# M6 — Nonlinear/slaving

\- multiple perturbation trajectories;  
\- slow/fast split;  
\- \\(b\\approx h(a)\\) evaluated out-of-sample.

\#\#\# M7 — Medium ConceptNet

\- \>=50k nodes;  
\- sparse memory-safe pipeline;  
\- iterative eigensolver;  
\- hierarchy/plateau metrics;  
\- runtime/memory report.

\---

\#\# 30\. Synthetic validation

До сильных утверждений по ConceptNet обязательны:

1\. hierarchical SBM;  
2\. nested weighted blocks;  
3\. graph with planted slow spectral modes;  
4\. graph without timescale separation;  
5\. hub-heavy sparse graph;  
6\. optional graph with planted nonlinear slow manifold.

Метод должен восстанавливать planted macrostructure и не находить plateaus там, где они не заложены.

\---

\#\# 31\. Computational constraints

Использовать sparse tooling:

\- \\\`scipy.sparse\\\`;  
\- \\\`eigsh\\\`;  
\- \\\`lobpcg\\\`;  
\- ARPACK или эквивалент.

Не строить dense \\(N\\times N\\) для medium/full ConceptNet.

Логировать wall time, peak RAM, solver convergence и cache status.

\---

\#\# 32\. Numerical stability

Обрабатывать:

\- disconnected components;  
\- zero-degree nodes;  
\- near-degenerate eigenvalues;  
\- hub-localized modes;  
\- eigenvector sign ambiguity;  
\- eigenspace rotation.

При cross-scale comparison сравнивать \*\*subspaces\*\*, а не отдельные eigenvectors.

Использовать principal angles, projection-matrix distance и Procrustes alignment.

\---

\#\# 33\. ConceptNet confounds

Обязательные ablations:

\- \\\`RelatedTo\\\` dominance;  
\- \\\`FormOf\\\` inclusion/exclusion;  
\- relation-specific graphs;  
\- directed vs symmetrized;  
\- raw vs transformed weights;  
\- high-degree hubs;  
\- source/dataset effects;  
\- sense-specific nodes;  
\- largest-component filtering.

\---

\#\# 34\. Graphon acceptance rules

Graphon result допускается в отчёт только если явно указано:

1\. node ordering / latent coordinate;  
2\. block masses;  
3\. kernel normalization;  
4\. dense/rescaled-sparse/sampled interpretation;  
5\. relabeling invariance treatment;  
6\. alignment between scales.

Не сравнивать graphon images naive pixel-wise без alignment.

\---

\#\# 35\. Основные outputs

\- \*\*O1:\*\* \\(G\_0\\to G\_1\\to\\cdots\\to G\_S\\).  
\- \*\*O2:\*\* order-parameter profile \\(r\_0,r\_1,\\ldots,r\_S\\).  
\- \*\*O3:\*\* cross-scale slow-subspace distance matrix.  
\- \*\*O4:\*\* semantic scale plateau candidates.  
\- \*\*O5:\*\* Haken-vs-baseline matched-compression table.  
\- \*\*O6:\*\* slaving evidence.  
\- \*\*O7:\*\* graphon sequence \\(W\_0,W\_1,\\ldots,W\_S\\).

\---

\#\# 36\. Falsification criteria

Гипотеза Haken-coarsening считается неподдержанной, если устойчиво наблюдается одно или несколько:

1\. нет timescale separation;  
2\. \\(r\_s\\) хаотично меняется;  
3\. slow subspace нестабилен к малым perturbations;  
4\. Haken не лучше baselines при равном compression;  
5\. modes почти полностью объясняются degree/hubs;  
6\. fast modes не предсказываются slow modes;  
7\. semantic coherence исчезает после контроля topology;  
8\. graphon plateau исчезает в sparse-control branch.

\*\*Negative result — валидный результат проекта.\*\*

\---

\#\# 37\. Success criteria

Минимальный publishable success:

1\. воспроизводимый алгоритм;  
2\. явный merge rule;  
3\. synthetic validation;  
4\. ConceptNet experiment;  
5\. \>=3 baselines;  
6\. \>=2 null models;  
7\. measurable cross-scale stability;  
8\. честный вывод о поддержке/неподдержке Haken assumptions.

Strong success:

\- nontrivial plateau;  
\- persistent slow subspace;  
\- Haken лучше baselines по dynamic preservation;  
\- semantically coherent coarse blocks;  
\- slaving generalizes out-of-sample.

\---

\#\# 38\. Стартовый config

\\\`\\\`\\\`yaml  
dataset:  
  source: conceptnet-5.7  
  language: en  
  relations:  
    \- RelatedTo  
    \- IsA  
    \- PartOf  
    \- HasA  
    \- UsedFor  
    \- HasProperty  
  min\_weight: 1.0  
  max\_nodes: 10000  
  component: largest

graph:  
  directed: false  
  weight\_transform: log1p  
  operator: normalized\_adjacency

dynamics:  
  model: linear  
  alpha: 1.0  
  beta: auto\_critical  
  top\_k\_modes: 64

order\_parameters:  
  selection:  
    \- eigengap  
    \- timescale\_gap  
    \- bootstrap\_stability  
  max\_r: 32

coarsening:  
  method: agglomerative  
  target\_reduction\_per\_step: 0.5  
  min\_levels: 5  
  max\_levels: 12

evaluation:  
  perturbations: 32  
  bootstrap\_runs: 20  
  baselines:  
    \- random\_matching  
    \- heavy\_edge  
    \- leiden  
    \- spectral  
    \- sbm  
\\\`\\\`\\\`

\\\`beta: auto\_critical\\\` \= подобрать \\(\\beta\\), чтобы ведущие нетривиальные modes оказались близко к slow/critical regime при сохранении численной устойчивости. Правило подбора документировать и аблировать.

\---

\#\# 39\. Приоритет разработки

Строго предпочтительный порядок:

1\. ConceptNet parser;  
2\. deterministic sparse graph builder;  
3\. normalized operator;  
4\. spectrum/timescale diagnostics;  
5\. Haken embedding;  
6\. one-step quotient;  
7\. multi-scale loop;  
8\. plateau metrics;  
9\. baselines;  
10\. null models;  
11\. nonlinear/slaving;  
12\. graphon family modeling.

\*\*Не начинать с neural graphon model.\*\* Первый вопрос: существует ли вообще маломерная макродинамика?

\---

\#\# 40\. Формат научного отчёта

Каждый эксперимент должен содержать:

\#\#\# Observed  
Только измеренные результаты.

\#\#\# Interpretation  
Что они могут означать.

\#\#\# Alternative explanations  
Baselines/confounds/null hypotheses.

\#\#\# Status  
Одно из:

\- supports hypothesis;  
\- weakly supports;  
\- inconclusive;  
\- contradicts.

Не смешивать Observation и Interpretation.

\---

\#\# 41\. Ключевая схема

\\\[  
\\boxed{  
\\text{ConceptNet microstructure}  
\\rightarrow  
\\text{network dynamics}  
\\rightarrow  
\\text{slow collective modes}  
\\rightarrow  
\\text{order-parameter coordinates}  
\\rightarrow  
\\text{vertex equivalence}  
\\rightarrow  
\\text{quotient graph}  
\\rightarrow  
\\text{new scale}  
}  
\\\]

и затем повтор:

\\\[  
\\boxed{G\_0\\to G\_1\\to G\_2\\to\\cdots}  
\\\]

Проект НЕ спрашивает:

\> «Сколько кластеров есть в ConceptNet?»

Он спрашивает:

\> \*\*На каких масштабах микроскопические семантические различия можно удалить, не меняя существенно доминирующую коллективную динамику?\*\*

Это рабочее операционное определение \*\*Haken-coarsening\*\*.

\---

\#\# 42\. Методологическая база

Использовать как минимум:

\- Haken — synergetics, order parameters, slaving principle;  
\- Малинецкий, Потапов — параметры порядка и инерциальные многообразия;  
\- Lovász — graphons, kernels, graph limits;  
\- Wolfe & Olhede — nonparametric graphon estimation;  
\- Veitch & Roy / Borgs et al. — sparse limits / graphex;  
\- ConceptNet 5 documentation.

Связка

\\\[  
\\text{Haken}  
\\rightarrow  
\\text{iterative semantic coarsening}  
\\rightarrow  
\\text{graphon family}  
\\\]

является \*\*исследовательским синтезом\*\* и должна быть экспериментально проверена.  
