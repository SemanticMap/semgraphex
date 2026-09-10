# Дневник эксперимента M4.1 — Haken-coarsening synthetic falsification

Дата: 2026-09-10

## 1. Почему понадобился M4.1

Исходный M4 успешно выполнил вычислительный pipeline, но не обнаружил plateau ни в positive control, ни в negative control. При этом slow-subspace distance у positive control на части переходов был заметно ниже negative control. Рабочая гипотеза M4.1: исходный full-state trajectory error может быть слишком строгим для редукции Хакена, поскольку он штрафует различия именно в быстрых микроскопических степенях свободы, которые coarsening должен уметь отбрасывать.

Это не означает, что full-state metric удаляется. M4.1 сохраняет её как контроль и добавляет две независимые метрики.

## 2. Термины

- **Mode / мода** — характерный коллективный паттерн изменения состояния сети, соответствующий собственному вектору выбранного оператора.
- **Slow mode / медленная мода** — мода с большим временем релаксации, то есть затухающая медленнее других.
- **Order parameter candidate / кандидат в параметр порядка** — амплитуда выбранной медленной моды. В теории Хакена параметр порядка — малое число макропеременных, управляющих коллективным режимом. Здесь это пока кандидат, а не доказанный параметр порядка.
- **Transient / переходный процесс** — начальная кратковременная реакция после возмущения до выхода на более медленный режим.
- **Full-state trajectory error** — относительная ошибка между полной fine-trajectory и поднятой обратно в fine-space coarse-trajectory.
- **Slow-order-parameter error** — относительная ошибка только амплитуд медленных мод. Быстрые компоненты состояния в неё не входят.
- **Post-transient error** — full-state ошибка только на поздней части траектории после отбрасывания начального переходного процесса.
- **Sweep / параметрический прогон** — систематический перебор сетки параметров вместо выбора одного удобного значения.
- **Sensitivity / чувствительность** — доля positive controls, для которых detector правильно обнаружил plateau.
- **Specificity / специфичность** — доля negative controls, для которых detector правильно не обнаружил plateau.
- **False-positive rate** — доля negative controls, ошибочно признанных plateau.
- **Youden J** — sensitivity - false-positive-rate. 1 означает идеальное разделение, 0 — отсутствие дискриминации.

## 3. Что реализовано

### 3.1 Slow-order-parameter trajectory error

Для fine slow basis `Q` вычисляются амплитуды

`A_f(t) = X_f(t) Q`

и

`A_c(t) = L X_c(t) Q`,

где `L` — lifting, то есть перенос coarse-state обратно на исходные fine nodes.

Метрика:

`D_slow = ||A_f - A_c||_F / ||A_f||_F`.

Она инвариантна к общему повороту/смене знаков базиса `Q` и измеряет сохранность макродинамики внутри выбранного slow subspace.

### 3.2 Post-transient trajectory error

При `post_transient_start_fraction = 0.5` используется только вторая половина временной сетки:

`D_tail = ||X_f(t>=t*) - L X_c(t>=t*)||_F / ||X_f(t>=t*)||_F`.

Дополнительно сохраняется `post_transient_energy_fraction` — доля нормы fine trajectory, оставшаяся после `t*`. Она нужна, чтобы не интерпретировать огромную относительную ошибку как содержательный результат, когда сама динамика к этому времени почти полностью затухла.

### 3.3 Сетка M4.1

Positive control варьируется по:

- bridge weight: 0.02, 0.05, 0.10;
- target reduction per step: 0.35, 0.50;
- seed: 1729, 1730.

Итого 12 positive runs.

Negative control не имеет слабых межблочных мостов как параметра, поэтому не дублируется по bridge weight; он прогоняется по двум compression settings и двум seeds. Итого 4 negative runs.

Detector sweep перебирает:

- trajectory metric: full, slow, post_transient;
- r tolerance: 0, 1;
- subspace threshold: 0.20, 0.45, 0.80;
- trajectory threshold: 0.25, 0.50, 0.80, 1.20;
- eigenvalue threshold: 0.15, 0.30, 0.60.

Минимальная длина plateau: 2 последовательных перехода.

## 4. Предзарегистрированный критерий до просмотра результата

Чтобы не считать один удачно подобранный threshold доказательством, заранее задано:

- sensitivity >= 0.75;
- specificity >= 0.75;
- этим условиям должны удовлетворять минимум 3 разных detector settings.

Статусы:

- `candidate_discriminative_region` — выполнено условие выше;
- `isolated_discriminative_settings` — есть отдельные хорошие thresholds, но их меньше трёх;
- `no_discriminative_setting` — даже отдельного разделяющего threshold на заданной сетке нет.

## 5. Изменения кода

Добавлены:

- `src/semmap_haken/m41_metrics.py` — три trajectory metrics и adjacent-scale recomputation;
- `src/semmap_haken/m41_sweep.py` — parameter sweep, detector sweep, sensitivity/specificity/Youden J;
- `src/semmap_haken/m41_cli.py` — CLI запуска;
- `configs/m41_synthetic_sweep.yaml` — предзаданная сетка эксперимента;
- `tests/test_semmap_haken_m41.py` — проверки семантики новых метрик;
- `.github/workflows/haken_colab_smoke.yml` — автоматический запуск M4.1 после тестов.

## 6. Вычислительный запуск

GitHub Actions run: `34501256901`.

На момент создания этой записи run запущен; результаты будут дописаны ниже после завершения без изменения предзарегистрированных thresholds.

## 7. Результаты

_Заполняется после завершения run._

## 8. Интерпретация

_Заполняется после результатов._

## 9. Альтернативные объяснения и ограничения

_Заполняется после результатов._
