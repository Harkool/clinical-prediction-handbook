# Clinical Prediction Handbook: Python practice companion

**English** · [中文](README.md)

Companion to the *Clinical Prediction Model Development Practical Handbook*, version **v2.0.0**. All implementations are Python; R is not required. The book contains practical steps, formulas and interpretation, with a separate short linear algebra primer. Code and commands live here.

All **31 original chapters have executable entry points**, including 16 numerical labs. The [chapter coverage inventory](docs/chapter-coverage.md) distinguishes implemented exercises, partial coverage and manual work; numbered coverage does not imply every method is implemented.

The separate linear algebra primer has a small numerical entry point: `python run.py foundations --mode quick`, covering dot products, rank, OLS residual orthogonality and ridge shrinkage without renumbering the original 31 chapters.

Case A is a frozen synthetic cohort of 3,200 adults, 384 events, 14 raw predictors and 18 non-intercept parameters. The target is CD ≥ III, including death, within 30 days from surgery start. Prediction time is distinct from outcome time origin. No real patient data are included.

## Getting started

Python 3.11 or 3.12 is recommended; the minimum compatible version is 3.9. See [installation](docs/installation.md).

```sh
git clone https://github.com/Harkool/clinical-prediction-handbook.git
cd clinical-prediction-handbook
python -m pip install -r requirements.txt
python run.py doctor
python run.py chapter 08 --mode quick
python run.py verify --mode quick
```

A chapter prepares its required shared data and model automatically. Invoke the actual `run.py` path to work from another directory; relative `--input`, `--schema` and `--output` paths remain relative to your current directory.

```sh
python run.py chapter 05 --mode quick
python run.py chapter 24 --mode quick
python run.py course --mode quick       # chapters 01–20
python run.py math all --mode quick     # 16 labs for chapters 21–31
python run.py verify --mode quick
```

`chapter 21` through `chapter 31` run every lab registered to that chapter. `math 24b` selects one lab, while `math 24` selects both chapter 24 labs.

## Baseline characteristics: Table 1

```sh
python run.py chapter 03 --mode quick
python run.py baseline --mode quick
python run.py baseline --input examples/reference/main_case.csv \
  --schema templates/table1_schema.json --output outputs/my-table1
```

The [schema](templates/table1_schema.json) explicitly defines variables, types, units, category levels, grouping and summaries. The default table describes all 14 raw predictors overall and by outcome: mean (sample SD), median [quartiles], or count/observed denominator (%). Missing values and zero-count declared levels remain visible.

Exports include numeric and display CSV, Markdown, HTML, a LaTeX fragment and a status JSON. Category percentages use observed denominators; missing percentages use total denominators. This standard-library module performs no imputation or univariable predictor selection. SD with one observation is not estimable, not zero. See [Table 1 details](docs/baseline-table.md).

## Modular Python analysis toolkit

All 12 capabilities in the supplied `Codebase.py` have been rewritten into separate modules: preprocessing, CV, learning curves, confusion matrices, ROC, PR, DCA, calibration, paired DeLong comparison, model comparison and patient-ID-bound probability storage. See the [migration mapping and toolkit guide](docs/python-toolkit.md). Legacy class names and signatures are not retained as the new API.

```sh
python -m pip install -r requirements-toolkit.txt
python run.py toolkit --mode quick
```

The demo writes Table 1, fixed holdout predictions and model comparisons, six plots, OOF predictions, fold metrics and learning curves. Preprocessing is fitted within training folds; threshold 0.15 is prespecified; comparisons require aligned patient IDs. Saved probabilities cannot predict new patients. Fold SD on learning curves is descriptive, not a confidence interval.

## Modes and provenance

| Resampling budget | quick | book |
|---|---:|---:|
| Development bootstrap | 20 | 200 |
| Fixed-model apparent metrics | 40 | 500 |
| Calibration development / conditional | 20 / 40 | 200 / 500 |
| Double bootstrap outer × inner | 10 × 10 | 200 × 50 |
| Descriptive subgroup resampling | 40 | 300 |

```sh
python run.py course --mode book
python run.py verify --mode book
python run.py math all --mode book
```

Outputs are separated into `outputs/quick/` and `outputs/book/`. Caches bind inputs, complete prediction rules, sources, dependencies, RNG scheme and budgets. Changed provenance marks older chapters `STALE`; failed reruns do not retain old `PASS` reports. `verify` independently reconstructs all 3,200 predictions, apparent C and Brier, and checks recorded chapter/source freshness; it does not claim that every chapter has run.

Python fits the same frozen inputs as the historical book, enabling coefficient/prediction comparison with [historical references](examples/reference/README.md). New simulation and resampling use PCG64; equal budgets do not imply identical intervals to the former R RNG. Development uncertainty, conditional fixed-model intervals and pointwise calibration bands are separately described. See [validation](docs/validation.md).

## Scope and checks

Chapter 7 includes a training-side imputation exercise, not full MICE. Chapter 9 implements cumulative incidence and cause-specific Cox demonstrations; **Fine–Gray regression and its validation are not implemented**. Chapters 10–11 compare Logistic and ridge with inner tuning; forests, boosting and TabPFN are not integrated into the main case, although some independent math labs simulate them. Chapter 14 uses independent synthetic validation/update data. Reporting, submission and human review retain `PENDING` where required.

| Directory | Purpose |
|---|---|
| `cph/` | Python case lessons, validation, baseline tables and reusable tools |
| `math/` | 16 numerical labs for chapters 21–31 |
| `templates/` | Table schemas and human review materials |
| `examples/reference/` | Frozen synthetic inputs and attributed historical numeric references |
| `outputs/` | Current results, status, logs and plots; ignored by Git |
| `tests/` | Numerical, negative-input, patient-separation and failure-status tests |

```sh
python run.py course --mode quick
python run.py verify --mode quick
python -m unittest discover -s tests -v
```

GitHub Actions check Windows, macOS and Linux; full budgets are manually triggered. [Chapter guide](docs/chapter-guide.md) · [Learning path](docs/learning-path.md) · [Math labs](docs/math-labs.md) · [Troubleshooting](docs/troubleshooting.md) · [Contributing](CONTRIBUTING.md)

Code, repository documentation, templates and synthetic examples use [MIT](LICENSE). Book manuscripts and PDFs are separately licensed; see [book licensing](BOOK_LICENSE.md). Cite the software version using [CITATION.cff](CITATION.cff).
