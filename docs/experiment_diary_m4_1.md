# Дневник эксперимента M4.1 — Haken-coarsening synthetic falsification

Дата: 2026-09-10

## 1. Исходная проблема

Исходный M4 успешно выполнил вычислительный pipeline, но не обнаружил plateau ни в positive control, ни в negative control. При этом slow-subspace distance у positive control на части переходов был заметно ниже negative control.

Рабочая гипотеза M4.1 была следующей: исходный `full-state trajectory error` может быть слишком строгим для Haken-coarsening, поскольку он штрафует различия в быстрых микроскопических степенях свободы, хотя цель редукции Хакена — сохранить прежде всего медленную макродинамику.

Чтобы не подменять исходный критерий более удобным после просмотра результата, M4.1 **не удаляет** старую метрику. Вместо этого одновременно сохраняются три разных измерения динамической ошибки.

## 2. Термины

- **Mode / мода** — характерный коллективный паттерн изменения состояния сети, соответствующий собственному вектору выбранного оператора.
- **Slow mode / медленная мода** — мода с большим временем релаксации, то есть затухающая медленнее других.
- **Relaxation time / время релаксации** — характерное время, за которое возмущение данной моды заметно затухает.
- **Order parameter / параметр порядка** — в синергетике Хакена небольшое число макропеременных, описывающих коллективный режим системы.
- **Order-parameter candidate / кандидат в параметр порядка** — в текущем эксперименте амплитуда выбранной медленной спектральной моды. Термин `candidate` принципиален: сам факт медленности ещё не доказывает, что мода является параметром порядка в строгом смысле Хакена.
- **Fine graph** — граф на более детальном уровне до очередного сжатия.
- **Coarse graph** — граф после объединения нескольких fine-вершин в супервершины.
- **Lifting** — перенос состояния coarse graph обратно в пространство fine graph: каждой исходной вершине присваивается состояние её супервершины.
- **Slow subspace / медленное подпространство** — пространство, натянутое на выбранные медленные моды. Оно описывает набор макроскопических направлений динамики, а не одну конкретную моду.
- **Transient / переходный процесс** — начальная кратковременная реакция после возмущения до выхода на более медленный режим.
- **Full-state trajectory error** — относительная ошибка между полной fine-trajectory и поднятой обратно в fine-space coarse-trajectory.
- **Slow-order-parameter error** — относительная ошибка только амплитуд медленных мод. Быстрые компоненты состояния в неё не входят.
- **Post-transient error** — full-state ошибка только на поздней части траектории после отбрасывания начального переходного процесса.
- **Sweep / параметрический прогон** — систематический перебор сетки параметров вместо выбора одного удобного значения.
- **Positive control / положительный контроль** — синтетический граф, в который намеренно заложена макроструктура с разделением масштабов; алгоритм должен уметь её обнаруживать.
- **Negative control / отрицательный контроль** — синтетический граф без специально заложенной блочной макроструктуры; алгоритм не должен находить в нём аналогичное plateau.
- **Sensitivity / чувствительность** — доля positive controls, для которых detector правильно обнаружил plateau.
- **Specificity / специфичность** — доля negative controls, для которых detector правильно не обнаружил plateau.
- **False-positive rate / доля ложноположительных срабатываний** — доля negative controls, ошибочно признанных содержащими plateau.
- **Youden J** — `sensitivity - false-positive-rate`, эквивалентно `sensitivity + specificity - 1`. Значение 1 означает идеальное разделение positive/negative, 0 — отсутствие полезного различения.
- **Gate / критерий допуска** — отдельное условие, которое переход между масштабами обязан выполнить, чтобы считаться частью plateau.
- **Target reduction** — доля вершин, которую алгоритм пытается убрать за один шаг coarsening. Например, `0.35` означает попытку уменьшить число вершин примерно на 35% за шаг.
- **Bridge weight** — вес слабых связей между заранее заданными макроблоками positive control. Чем он меньше относительно внутренних связей, тем сильнее разделены блоки.
- **Seed** — начальное число генератора псевдослучайных чисел. Повторение эксперимента с тем же seed должно воспроизводить те же случайные возмущения и решения при равных условиях.

## 3. Новые метрики M4.1

### 3.1 Full-state trajectory error — контрольная старая метрика

Она оставлена без замены:

`D_full = ||X_f - L X_c||_F / ||X_f||_F`,

где:

- `X_f` — fine trajectory;
- `X_c` — coarse trajectory;
- `L` — lifting coarse trajectory обратно на fine nodes;
- `||.||_F` — норма Фробениуса, то есть корень из суммы квадратов всех ошибок по времени и вершинам.

Эта метрика спрашивает: «насколько coarse system воспроизводит всё микроскопическое состояние?»

### 3.2 Slow-order-parameter trajectory error

Для fine slow basis `Q` вычисляются амплитуды:

`A_f(t) = X_f(t) Q`,

`A_c(t) = L X_c(t) Q`.

Метрика:

`D_slow = ||A_f - A_c||_F / ||A_f||_F`.

Она спрашивает более узко: «сохранилась ли динамика внутри выбранного slow subspace?»

Это ближе к идее Хакена, поскольку быстрые компоненты состояния не обязаны точно воспроизводиться после редукции.

### 3.3 Post-transient trajectory error

При `post_transient_start_fraction = 0.5` используется только вторая половина временной сетки:

`D_tail = ||X_f(t>=t*) - L X_c(t>=t*)||_F / ||X_f(t>=t*)||_F`.

Дополнительно сохраняется `post_transient_energy_fraction` — доля нормы fine trajectory, оставшаяся после `t*`. Это нужно, чтобы не переинтерпретировать относительную ошибку, когда сама динамика к позднему времени почти полностью затухла.

## 4. Сетка эксперимента

Positive control варьировался по:

- bridge weight: `0.02`, `0.05`, `0.10`;
- target reduction per step: `0.35`, `0.50`;
- seed: `1729`, `1730`.

Итого: **12 positive runs**.

Negative control не имеет слабых межблочных мостов как параметра, поэтому не дублировался по bridge weight. Он прогонялся по двум значениям target reduction и двум seeds.

Итого: **4 negative runs**.

Detector sweep перебирал:

- trajectory metric: `full`, `slow`, `post_transient`;
- `r tolerance`: `0`, `1`;
- subspace threshold: `0.20`, `0.45`, `0.80`;
- trajectory threshold: `0.25`, `0.50`, `0.80`, `1.20`;
- eigenvalue threshold: `0.15`, `0.30`, `0.60`.

Минимальная длина plateau: **2 последовательных перехода**.

`r tolerance` — допустимое изменение числа выбранных slow modes между соседними масштабами. При `0` число должно оставаться в точности тем же.

`subspace threshold` — максимально допустимое изменение slow subspace после сжатия.

`eigenvalue threshold` — максимально допустимое изменение slow eigenvalues; eigenvalue здесь характеризует масштаб/скорость действия соответствующей моды выбранного оператора.

## 5. Предзарегистрированный критерий до просмотра результата

Чтобы один удачно подобранный threshold не считался доказательством, заранее было задано:

- sensitivity >= `0.75`;
- specificity >= `0.75`;
- этим условиям должны удовлетворять минимум **3 разных detector settings**.

Статусы:

- `candidate_discriminative_region` — есть как минимум три разных набора порогов, устойчиво разделяющих positive и negative controls;
- `isolated_discriminative_settings` — есть отдельные хорошие thresholds, но их меньше трёх;
- `no_discriminative_setting` — на заданной сетке нет настройки, удовлетворяющей предзарегистрированному условию.

## 6. Реализация

Добавлены:

- `src/semmap_haken/m41_metrics.py` — три trajectory metrics и adjacent-scale recomputation;
- `src/semmap_haken/m41_sweep.py` — parameter sweep, detector sweep, sensitivity/specificity/Youden J;
- `src/semmap_haken/m41_cli.py` — CLI запуска;
- `src/semmap_haken/m41_diagnose.py` — post-hoc диагностика причин отказа **без изменения thresholds**;
- `configs/m41_synthetic_sweep.yaml` — предзаданная сетка эксперимента;
- `tests/test_semmap_haken_m41.py` — проверки математического смысла новых метрик;
- `.github/workflows/haken_colab_smoke.yml` — автоматический запуск M4.1.

Проверки новых метрик:

1. slow metric не реагирует на ошибку, лежащую строго вне slow subspace, тогда как full metric её видит;
2. post-transient metric не учитывает ошибку, существующую только в отброшенной ранней части trajectory;
3. adjacent fine-to-coarse mapping корректно восстанавливается из сохранённой ancestry исходных вершин;
4. detector может использовать slow metric, не переопределяя full metric.

Все четыре теста прошли.

## 7. Вычислительные запуски

### 7.1 Основной M4.1 sweep

GitHub Actions run: `34501256901`.

Результат workflow: **success**.

Внутри run успешно выполнены:

- установка зафиксированного Python-окружения;
- unit tests M4.1;
- повтор исходного M4;
- M4.1 sweep;
- генерация отчёта;
- сохранение evidence artifact.

### 7.2 Диагностический повтор без retuning

GitHub Actions run: `34501546335`, commit `c3140a99e2148b834df59339891bb61b081da466`.

Результат workflow: **success**.

Диагностический шаг не менял ни одного threshold. Он только разложил уже полученный результат по bridge weight, target reduction, seed и причинам отказа отдельных gates.

Evidence artifact: `haken-m41-evidence`, artifact ID `10162060261`, 230 файлов, ZIP SHA-256 `11ffe5d63d0876cc4df4ccc7a3c1e13316e2229fcf5b45381e593549284caaf8`.

## 8. Основной результат M4.1

Исходный M4 в том же окружении снова дал:

- positive control: `0` plateau candidates;
- negative control: `0` plateau candidates.

M4.1 sweep дал:

- positive runs: **12**;
- negative runs: **4**;
- robust detector settings: **0**;
- итоговый статус: **`no_discriminative_setting`**.

Следовательно, предзарегистрированный научный критерий M4.1 **не пройден**.

### 8.1 Лучший detector для каждой trajectory metric

| trajectory metric | sensitivity | specificity | false-positive rate | Youden J | r tolerance | subspace threshold | trajectory threshold | eigenvalue threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 0.500 | 1.000 | 0.000 | 0.500 | 0 | 0.45 | 0.50 | 0.15 |
| slow | 0.500 | 1.000 | 0.000 | 0.500 | 0 | 0.45 | 0.25 | 0.15 |
| post-transient | 0.500 | 1.000 | 0.000 | 0.500 | 0 | 0.45 | 0.50 | 0.15 |

То есть все три варианта дали одинаковую итоговую классификационную способность: правильно отвергнуты все negative controls, но найдено только 6 из 12 positive controls.

При этом slow metric допускает более строгий собственный threshold (`0.25` вместо `0.50`) и всё равно достигает той же чувствительности. Это показывает, что slow amplitudes действительно воспроизводятся численно лучше, чем всё состояние, но само по себе это не улучшило итоговую способность detector находить plateau.

## 9. Главный диагностический результат: причина 50% sensitivity

Разбиение positive detection rate по экспериментальным факторам показало для **всех трёх** trajectory metrics одну и ту же картину:

### По target reduction

- `target_reduction = 0.35`: detection rate = **1.000**;
- `target_reduction = 0.50`: detection rate = **0.000**.

### По bridge weight

- `0.02`: detection rate = `0.500`;
- `0.05`: detection rate = `0.500`;
- `0.10`: detection rate = `0.500`.

### По seed

- `1729`: detection rate = `0.500`;
- `1730`: detection rate = `0.500`.

Следовательно, наблюдаемая чувствительность `0.5` практически полностью объясняется **агрессивностью одного шага coarsening**, а не bridge weight или случайным seed.

Иными словами:

> При удалении примерно 35% вершин за шаг алгоритм стабильно сохраняет заложенную макроструктуру positive control; при попытке удалить примерно 50% вершин за шаг эта макроструктура перестаёт удовлетворять критериям plateau.

Это наиболее содержательный результат M4.1.

## 10. Поведение числа slow modes r

Для всех positive runs с `target_reduction=0.35`:

`r: 3 -> 3 -> 3 -> 3 -> 3`.

То есть число candidate order parameters остаётся стабильным на всех пяти рассматриваемых масштабах.

Для positive runs с `target_reduction=0.50`:

`r: 3 -> 3 -> 3 -> 3 -> 1`.

Последний агрессивный шаг приводит к collapse числа slow-mode candidates с трёх до одного.

Для negative control при reduction `0.35`:

`r: 6 -> 6 -> 6 -> 6 -> 6`,

но при этом slow subspace меняется существенно сильнее, поэтому одной стабильности числа `r` недостаточно для ложного обнаружения plateau.

Для negative control при reduction `0.50` наблюдаются нестабильные последовательности, например:

`6 -> 6 -> 6 -> 5 -> 2`

и

`6 -> 6 -> 5 -> 6 -> 1`.

Это подтверждает необходимость сравнивать не только число modes, но и само slow subspace.

## 11. Средние cross-scale показатели по отдельным run

### Positive control, reduction = 0.35

Во всех шести комбинациях bridge/seed detector находит plateau для full/slow/post-transient metrics.

Типичные средние значения по четырём переходам:

- `D_full`: примерно `0.324–0.409`;
- `D_slow`: примерно `0.103–0.153`;
- `D_tail`: примерно `0.256–0.300`;
- `D_subspace`: примерно `0.264–0.266`;
- mean eigenvalue error: примерно `0.265–0.269`;
- `r`: неизменно `3`.

Здесь `D_subspace` — расстояние между fine slow subspace и поднятым coarse slow subspace: чем оно меньше, тем ближе макродинамические направления двух масштабов.

### Positive control, reduction = 0.50

Ни одна из шести комбинаций bridge/seed не проходит detector:

- `D_full`: примерно `0.551–0.573`;
- `D_slow`: примерно `0.342–0.442`;
- `D_tail`: примерно `0.455–0.496`;
- mean `D_subspace`: примерно `0.380–0.517`;
- `r` в конце падает `3 -> 1`.

### Negative control, reduction = 0.35

Ни один run не классифицирован как plateau:

- `D_full`: `0.376–0.418`;
- `D_slow`: `0.176–0.208`;
- `D_tail`: `0.282–0.289`;
- `D_subspace`: `0.587–0.588`;
- `r`: стабильно `6`.

Это особенно важное наблюдение: по trajectory error negative control местами выглядит не хуже positive, но его slow subspace существенно менее устойчив. Поэтому detector отвергает его за счёт совокупности признаков, а не одной метрики.

### Negative control, reduction = 0.50

Также ни одного plateau:

- `D_full`: `0.514–0.548`;
- `D_slow`: `0.285–0.345`;
- `D_tail`: `0.368–0.399`;
- `D_subspace`: `0.506–0.732`;
- `r` заметно нестабилен.

## 12. Какие gates чаще всего не проходили

Под лучшими настройками detector были посчитаны причины отказа каждого перехода.

Важно: один переход может одновременно провалить несколько gates, поэтому суммы ниже могут быть больше числа переходов.

### Positive controls

Для `full`:

- eigenvalue distortion: 30;
- trajectory distortion: 19;
- subspace distortion: 11;
- r instability: 6.

Для `slow`:

- eigenvalue distortion: 30;
- trajectory distortion: 15;
- subspace distortion: 11;
- r instability: 6.

Для `post-transient`:

- eigenvalue distortion: 30;
- trajectory distortion: 12;
- subspace distortion: 11;
- r instability: 6.

### Negative controls

Для `full`:

- subspace distortion: 12;
- r instability: 5;
- trajectory distortion: 5;
- eigenvalue distortion: 4.

Для `slow`:

- subspace distortion: 12;
- trajectory distortion: 7;
- r instability: 5;
- eigenvalue distortion: 4.

Для `post-transient`:

- subspace distortion: 12;
- r instability: 5;
- eigenvalue distortion: 4;
- trajectory distortion: 1.

### Что это означает

Новые trajectory metrics действительно уменьшают число trajectory-gate failures у positive control:

- full: 19;
- slow: 15;
- post-transient: 12.

Однако главным ограничением positive control оказывается не только trajectory metric: очень часто нарушается eigenvalue gate, а при агрессивном reduction возникают также subspace distortion и падение `r`.

У negative control наиболее устойчивым препятствием является именно **subspace distortion**: 12 отказов для каждой trajectory metric. Это полезно, потому что slow-subspace criterion действительно отличает отсутствие устойчивой макроструктуры даже там, где отдельная trajectory metric выглядит приемлемо.

## 13. Наблюдаемое

1. M4.1 вычислительно воспроизводим и проходит unit tests.
2. Добавление slow и post-transient metrics не создаёт ложных plateau в negative control.
3. Все negative controls правильно отвергнуты: specificity = `1.0`.
4. Общая sensitivity = `0.5`, поэтому предзарегистрированный robust criterion не выполнен.
5. Все positive cases с reduction `0.35` обнаружены; все с reduction `0.50` отвергнуты.
6. Bridge weight в диапазоне `0.02–0.10` и два проверенных seeds не объясняют различие результатов.
7. При reduction `0.35` positive slow dimension сохраняется `3 -> 3 -> 3 -> 3 -> 3`.
8. При reduction `0.50` positive slow dimension заканчивается collapse `3 -> 1`.
9. Slow trajectory error существенно ниже full-state error в positive control при умеренном reduction.
10. Negative control при reduction `0.35` может сохранять число `r=6`, но не сохраняет само slow subspace: `D_subspace` около `0.59` против примерно `0.265` у successful positive runs.

## 14. Интерпретация

Наиболее осторожная интерпретация:

> M4.1 пока **не подтверждает** универсальную устойчивость Haken-coarsening, но показывает существование режима умеренного coarsening, в котором planted hierarchical positive control устойчиво проходит многомасштабные критерии, тогда как homogeneous negative control их не проходит.

Это отличается от исходного M4, где единственный заранее выбранный режим `50%` reduction фактически оказался слишком агрессивным.

С научной точки зрения появляется более содержательная гипотеза:

`существует критическая скорость/шаг coarsening, выше которой медленная макроструктура разрушается`.

Или, если обозначить долю удаления вершин за шаг через `c`, может существовать диапазон

`c < c_crit`,

в котором slow subspace остаётся устойчивым, и

`c > c_crit`,

в котором начинается потеря макродинамической структуры.

M4.1 пока не определяет `c_crit`; он лишь показывает резкий контраст между `0.35` и `0.50` в данной синтетической модели.

## 15. Почему результат нельзя пока считать подтверждением основной гипотезы

1. Проверены только два значения reduction: `0.35` и `0.50`. Неизвестно, где проходит переход между устойчивым и неустойчивым режимом.
2. Всего два seeds недостаточны для статистической оценки устойчивости.
3. Positive generator сравнительно простой и вручную сконструирован.
4. Detector использует набор thresholds; хотя они не подгонялись после результата M4.1, требуется более широкая sensitivity analysis.
5. Выбор `r` пока основан на eigengap/timescale heuristic без bootstrap confidence.
6. Линейная динамика `dx/dt=(-alpha I + beta S)x` является модельным экспериментом; это ещё не доказательство истинной Haken dynamics семантической сети.
7. `slow modes` пока являются только кандидатами в order parameters: slaving relation для быстрых переменных ещё не проверена.
8. Средние eigenvalue errors могут превышать threshold, хотя отдельные последовательные переходы образуют plateau; поэтому для интерпретации нужно использовать transition-level, а не только run-average показатели.
9. Post-transient metric может становиться менее информативной, если к позднему времени почти вся динамическая энергия уже затухла; поэтому сохраняется energy fraction.

## 16. Альтернативные объяснения

### A. Эффект геометрии matching, а не Хакена

Успех при `0.35` может возникать потому, что greedy pair matching просто лучше работает при умеренном числе merges. Для исключения этого нужны matched-compression baselines: heavy-edge, random и обычный spectral coarsening.

### B. Артефакт positive generator

Ring/chord block construction может быть слишком согласована со spectral method. Нужны другие positive controls: hierarchical SBM и сети с явно заданным slow operator spectrum.

### C. Дискретный эффект размера

При reduction `0.50` граф быстро становится маленьким, и поздний collapse `r` может быть связан с конечным размером, а не с настоящей потерей масштаба. Нужно повторить на больших synthetic graphs.

### D. Неправильный invariant для eigenvalues

Raw eigenvalue preservation может быть слишком строгим между quotient scales. Возможно, важнее сохранять timescale ratios или slow generator spectrum после подходящей renormalization. Это требует отдельной теоретической проверки, а не простого ослабления threshold.

## 17. Статус M4.1

**Статус: INCONCLUSIVE, но обнаружен воспроизводимый режимный эффект.**

Предзарегистрированный критерий `sensitivity >= 0.75`, `specificity >= 0.75` минимум на трёх detector settings **не выполнен** по всей исходной сетке, потому что половина positive runs использует слишком агрессивный `target_reduction=0.50`.

Одновременно внутри заранее заданной сетки обнаружено устойчивое разделение по режиму coarsening:

- при reduction `0.35`: positive detection = `1.0`, negative detection = `0`;
- при reduction `0.50`: positive detection = `0`, negative detection = `0`.

Это **не основание переписать критерий задним числом**, но основание для нового заранее сформулированного эксперимента M4.2 по поиску критической величины шага coarsening.

## 18. Следующая проверяемая гипотеза для M4.2

До запуска ConceptNet разумно проверить:

> Для planted hierarchical networks существует интервал target reduction, в котором Haken-coarsening сохраняет slow subspace и candidate order parameters, тогда как при превышении критического reduction происходит резкий рост dynamic distortion и/или изменение slow dimension r.

Для этого следует заранее задать более плотную сетку, например:

`0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50`,

несколько размеров synthetic graph и существенно больше seeds.

Ключевая величина будущего эксперимента — не просто `plateau yes/no`, а функция

`D(c)`

ошибки/устойчивости от шага coarsening `c`, с поиском возможного knee/critical point.

**Knee / точка перегиба** — область, где небольшое дальнейшее увеличение степени сжатия начинает резко увеличивать потерю динамической структуры.
