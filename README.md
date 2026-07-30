# stock-simulation-system

> Single-source backtest framework. **Validation, not selection** — verify whether a strategy works, not which strategy to pick.

## What is this?

A backtest framework for evaluating quantitative stock-selection strategies. The whole system is built around two design choices that fall out from those goals:

1. **Stateless precomputation + thin stateful loop.** Score / filter / selection logic is computed once, up front. The day-by-day execution loop only reads the precomputed table. Look-ahead-bias is structurally impossible — the loop never re-scores on later data.
2. **CORE vs TACTICAL separation.** Cross-sectional precomputable sleeves (CORE) share the codebase with path-dependent single-name strategies (TACTICAL), but each gets a structurally different execution model.

## Features

- **Dual-sleeve architecture** — CORE (regular, precomputed) + TACTICAL (ad-hoc, live)
- **Bootstrap significance testing** — `VectorizedBootstrapEngine` + `ProcessPoolExecutor` for paired cross-world tests, with bull / bear / choppy regime classification
- **Pluggable scorers / filters / selectors** — every stage in `vector_calc` has an ABC base class; add a new strategy by dropping in a new file
- **Baseline strategies included** — momentum, mean reversion, MACD
- **Comprehensive evaluation** — CAGR, Sharpe, Sortino, drawdown, Calmar, IC, contribution, turnover
- **Real-data ready** — `yfinance` wrapper, no paid data sources required

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Zhigang2022/stock-simulation-system.git
cd stock-simulation-system

# 2. Set up a Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run a demo notebook
jupyter notebook notebooks/
```

## Architecture

See [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md) for the full architecture: signal pipeline, package structure, design principles.

![Pipeline](doc/pipeline.png)

## Project Structure

```
stock-simulation-system/
├── src2/                  # backtest engine
│   ├── states/            # stateful core (GlobalState)
│   ├── data_ingest/       # yfinance wrapper
│   ├── vector_calc/       # stateless precomputation
│   ├── iteration/         # day-by-day execution loop
│   ├── evaluation/        # metrics + significance + plots
│   └── bootstrap/         # synthetic worlds + paired tests
├── doc/                   # architecture documentation
│   ├── ARCHITECTURE.md
│   └── pipeline.png
├── notebooks/             # public demo notebooks
├── requirements.txt
├── LICENSE
└── README.md
```

## Status

This is the v0.1 release. CORE sleeve and bootstrap significance testing are stable. TACTICAL sleeve's ad-hoc strategy framework is in place; concrete strategies (MACD bull/bear cross being the first) are still being developed.

## License

MIT — see [LICENSE](LICENSE).
