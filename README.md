# InvestIQ

InvestIQ — это offline-first веб-приложение для учебного фундаментального анализа компаний S&P 500.

Пользователь может ввести тикер компании и получить:

- профиль компании
- исторические финансовые данные
- коэффициенты оценки и качества бизнеса
- сравнение с конкурентами
- локальную ML-классификацию стоимости компании
- простой учебный бэктест

Проект использует:

- FastAPI backend
- HTML/CSS/JavaScript frontend
- Plotly-графики

InvestIQ НЕ использует Streamlit.

---

# 1. Статус проекта

Текущая структура проекта:

\`\`\`text
investiq/
  backend/
    __init__.py
    app.py
    data_layer.py
    ratio_service.py
    peer_service.py
    ml_service.py
    backtest_service.py
    schemas.py

  frontend/
    index.html
    analysis.html
    style.css
    script.js

  scripts/
    build_dataset.py
    train_model.py

  data/
    raw/
      fundamentals.csv
      securities.csv
      prices-split-adjusted.csv
    sp500_complete.parquet
    sp500_prices.parquet

  models/
    valuation_model.joblib
    scaler.joblib
    model_metrics.json

  tests/
    conftest.py
    test_data_layer.py
    test_api.py
    test_ml_service.py

  requirements.txt
  README.md
\`\`\`

---

# 2. Важные правила проекта

Приложение разработано для автономной работы после локальной подготовки данных.

Во время работы backend НЕ обращается к:

- yfinance
- SEC EDGAR
- Wikipedia
- Kaggle API
- live market API
- любым сетевым источникам

Backend читает только локальные Parquet-файлы:

\`\`\`text
data/sp500_complete.parquet
data/sp500_prices.parquet
\`\`\`

Frontend использует Plotly через CDN в analysis.html.

Это означает, что графики требуют интернет-соединения, если Plotly не сохранён локально.

---

# 3. Требования к датасету

Используется Kaggle-датасет:

dgawlik/nyse

Необходимые файлы:

\`\`\`text
data/raw/fundamentals.csv
data/raw/securities.csv
data/raw/prices-split-adjusted.csv
\`\`\`

Игнорируемый файл:

\`\`\`text
data/raw/prices.csv
\`\`\`

---

# 4. Ограничения данных

Анализировать можно только тикеры, существующие в локально подготовленном датасете.

Некоторые популярные тикеры S&P 500 могут отсутствовать, если исходные CSV-файлы не содержат полных финансовых данных.

Итоговый датасет может содержать меньше 500 компаний. Это нормально, поскольку финальный набор зависит от пересечения:

- financial fundamentals
- securities metadata
- historical prices

---

# 5. Установка

## Шаг 1. Перейдите в папку проекта

\`\`\`bash
cd InvestIQ
\`\`\`

## Шаг 2. Создайте виртуальное окружение

### Mac/Linux

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

### Windows PowerShell

\`\`\`powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
\`\`\`

## Шаг 3. Установите зависимости

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Если requirements.txt отсутствует:

\`\`\`bash
pip install pandas pyarrow fastapi uvicorn pydantic scikit-learn joblib pytest httpx
\`\`\`

---

# 6. Подготовка данных

Создайте папку raw-data:

\`\`\`bash
mkdir -p data/raw
\`\`\`

Поместите CSV-файлы:

\`\`\`text
data/raw/fundamentals.csv
data/raw/securities.csv
data/raw/prices-split-adjusted.csv
\`\`\`

После этого создайте локальные Parquet-файлы:

\`\`\`bash
python scripts/build_dataset.py
\`\`\`

---

# 7. Обучение ML-модели

После подготовки Parquet-файлов:

\`\`\`bash
python scripts/train_model.py
\`\`\`

Будут созданы:

\`\`\`text
models/valuation_model.joblib
models/scaler.joblib
models/model_metrics.json
\`\`\`

Модель классифицирует компании как:

- UNDERVALUED
- FAIRLY_VALUED
- OVERVALUED

---

# 8. Запуск backend API

\`\`\`bash
uvicorn backend.app:app --reload
\`\`\`

Backend запускается по адресу:

\`\`\`text
http://127.0.0.1:8000
\`\`\`

Swagger docs:

\`\`\`text
http://127.0.0.1:8000/docs
\`\`\`

---

# 9. Запуск frontend

\`\`\`bash
open frontend/index.html
\`\`\`

Frontend ожидает backend по адресу:

\`\`\`javascript
const API_BASE = "http://localhost:8000/api";
\`\`\`

---

# 10. API endpoints

## Health

\`\`\`text
GET /api/health
\`\`\`

## Autocomplete

\`\`\`text
GET /api/autocomplete?q=AAPL
\`\`\`

## Company profile

\`\`\`text
GET /api/company_profile?ticker=AAPL
\`\`\`

## Historical financials

\`\`\`text
GET /api/historical_financials?ticker=AAPL&years=5
\`\`\`

## Financial ratios

\`\`\`text
GET /api/financial_ratios?ticker=AAPL
\`\`\`

## Peer comparison

\`\`\`text
GET /api/peer_comparison?ticker=AAPL&limit=8
\`\`\`

## ML classification

\`\`\`text
GET /api/ml_classify?ticker=AAPL
\`\`\`

## Backtest

\`\`\`text
POST /api/backtest
\`\`\`

Пример:

\`\`\`json
{
  "tickers": ["AAPL", "MSFT", "IBM", "CSCO"],
  "start_date": "2014-01-01",
  "end_date": "2016-12-31",
  "initial_capital": 10000
}
\`\`\`

---

# 11. Запуск тестов

\`\`\`bash
pytest -q
\`\`\`

Отдельные тесты:

\`\`\`bash
pytest -q tests/test_data_layer.py
pytest -q tests/test_api.py
pytest -q tests/test_ml_service.py
\`\`\`

---

# 12. GitHub setup

## Коммитить:

\`\`\`text
backend/
frontend/
scripts/
tests/
requirements.txt
README.md
\`\`\`

## Не коммитить:

\`\`\`text
data/raw/
data/*.parquet
models/*.joblib
\`\`\`

Рекомендуемый .gitignore:

\`\`\`gitignore
__pycache__/
*.pyc
.venv/
venv/
.env

data/raw/
data/*.parquet

models/*.joblib
models/model_metrics.json

.pytest_cache/
.coverage

.DS_Store
.vscode/
.idea/
\`\`\`

---

# 13. Git-команды

\`\`\`bash
git init
git status
git add backend frontend scripts tests requirements.txt README.md
git commit -m "Initial InvestIQ project"
git branch -M main
git push -u origin main
\`\`\`

---

# 14. Troubleshooting

## Dataset not found

\`\`\`bash
python scripts/build_dataset.py
\`\`\`

## Ticker not found

\`\`\`bash
python - <<'PY'
import pandas as pd

df = pd.read_parquet("data/sp500_complete.parquet")
print(sorted(df["ticker"].dropna().str.upper().unique())[:100])
PY
\`\`\`

## ModuleNotFoundError

\`\`\`bash
touch backend/__init__.py
\`\`\`

## Plotly charts do not show

Для полной offline-работы скачайте Plotly локально.

## Pydantic field_validator error

\`\`\`bash
pip install --upgrade fastapi pydantic
\`\`\`

---

# 15. Educational disclaimer

InvestIQ является учебным проектом.

Это НЕ инвестиционный совет.

ML-модель обучается на локальных данных и упрощённых правилах.

Бэктест является исключительно демонстрационным.
# InvestIQ

InvestIQ is an offline-first web application for educational fundamental analysis of S&P 500 companies.

It lets a user enter a ticker and review:

- company profile
- historical financials
- valuation and quality ratios
- peer comparison
- local ML valuation classification
- simple educational backtest

The project uses a **FastAPI backend** and a **HTML/CSS/JavaScript frontend** with Plotly charts.

InvestIQ does **not** use Streamlit.

---

## 1. Project status

Current project modules:

```text
investiq/
  backend/
    __init__.py
    app.py
    data_layer.py
    ratio_service.py
    peer_service.py
    ml_service.py
    backtest_service.py
    schemas.py

  frontend/
    index.html
    analysis.html
    style.css
    script.js

  scripts/
    build_dataset.py
    train_model.py

  data/
    raw/
      fundamentals.csv
      securities.csv
      prices-split-adjusted.csv
    sp500_complete.parquet
    sp500_prices.parquet

  models/
    valuation_model.joblib
    scaler.joblib
    model_metrics.json

  tests/
    conftest.py
    test_data_layer.py
    test_api.py
    test_ml_service.py

  requirements.txt
  README.md
```

---

## 2. Important project rules

The application is designed to work offline after local data preparation.

At runtime, the backend does **not** call:

- yfinance
- SEC EDGAR
- Wikipedia
- Kaggle API
- live market APIs
- any live network source

The backend reads only local Parquet files:

```text
data/sp500_complete.parquet
data/sp500_prices.parquet
```

The frontend uses Plotly from a CDN in `analysis.html`. This means the charts require internet access unless Plotly is downloaded and served locally.

---

## 3. Dataset requirements

### Main dataset

Use Kaggle dataset:

```text
dgawlik/nyse
```

Required files:

```text
data/raw/fundamentals.csv
data/raw/securities.csv
data/raw/prices-split-adjusted.csv
```

Ignored file:

```text
data/raw/prices.csv
```

---

## 4. Data limitations

Only tickers that exist in the prepared local dataset can be analyzed.

Some well-known S&P 500 tickers may not be available if the source CSV files do not contain complete usable fundamentals for them.

The current prepared dataset may have fewer than 500 companies. This is expected because the final dataset depends on the intersection of available fundamentals, securities metadata, and price data.

---

## 5. Installation

### Step 1: Go to the project folder

```bash
cd InvestIQ
```

### Step 2: Create a virtual environment

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist yet, install the packages manually:

```bash
pip install pandas pyarrow fastapi uvicorn pydantic scikit-learn joblib pytest httpx
```

Recommended `requirements.txt` content:

```text
pandas
pyarrow
fastapi
uvicorn
pydantic
scikit-learn
joblib
pytest
httpx
```

---

## 6. Data setup

Create the raw data folder:

```bash
mkdir -p data/raw
```

Place the required CSV files here:

```text
data/raw/fundamentals.csv
data/raw/securities.csv
data/raw/prices-split-adjusted.csv
```

Then build the local Parquet datasets:

```bash
python scripts/build_dataset.py
```

Expected output files:

```text
data/sp500_complete.parquet
data/sp500_prices.parquet
```

The build script prints a data quality report with:

- number of companies
- number of rows
- year range
- missing values before imputation
- missing values after imputation
- number of price rows

---

## 7. Train the ML valuation model

After building the Parquet files, train the model:

```bash
python scripts/train_model.py
```

Expected output files:

```text
models/valuation_model.joblib
models/scaler.joblib
models/model_metrics.json
```

The model classifies tickers as:

```text
UNDERVALUED
FAIRLY_VALUED
OVERVALUED
```

The model is trained from local financial ratios only.

It is not trained from live prices, analyst forecasts, or real investment outcomes.

---

## 8. Run the backend API

Start FastAPI:

```bash
uvicorn backend.app:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Stop the backend with:

```text
CTRL + C
```

---

## 9. Open the frontend

Open the home page directly in the browser:

```bash
open frontend/index.html
```

Or open the full path:

```bash
open frontend/index.html
```

The frontend expects the backend API to be running at:

```javascript
const API_BASE = "http://localhost:8000/api";
```

If the backend is not running, the frontend will not be able to load company analysis data.

---

## 10. API endpoints

### Health

```http
GET /api/health
```

Checks dataset availability and backend status.

### Autocomplete

```http
GET /api/autocomplete?q=AAPL
```

Returns matching local tickers and company names.

### Company profile

```http
GET /api/company_profile?ticker=AAPL
```

Returns latest company profile and core metrics.

### Historical financials

```http
GET /api/historical_financials?ticker=AAPL&years=5
```

Returns annual financial history.

### Financial ratios

```http
GET /api/financial_ratios?ticker=AAPL
```

Returns beginner-friendly ratio cards and financial health score.

### Peer comparison

```http
GET /api/peer_comparison?ticker=AAPL&limit=8
```

Returns target company, peer companies, averages, bar-chart data, and radar-chart data.

### ML classification

```http
GET /api/ml_classify?ticker=AAPL
```

Returns local valuation classification, probabilities, confidence, feature importance, and explanation.

The ML model is loaded lazily. It is not trained at API startup.

If model files are missing, the backend uses a rule-based fallback.

### Backtest

```http
POST /api/backtest
```

Example body:

```json
{
  "tickers": ["AAPL", "MSFT", "IBM", "CSCO"],
  "start_date": "2014-01-01",
  "end_date": "2016-12-31",
  "initial_capital": 10000
}
```

The backtest is educational only.

Strategy:

```text
Buy when latest available annual fundamentals show:
- P/E below 15
- ROE above 15%

Sell:
- after 252 trading days
- or at the selected end date
```

---

## 11. Run tests

Install test dependencies:

```bash
pip install pytest httpx
```

Run all tests:

```bash
pytest -q
```

Run specific test files:

```bash
pytest -q tests/test_data_layer.py
pytest -q tests/test_api.py
pytest -q tests/test_ml_service.py
```

The tests create temporary Parquet files with a small fake dataset.

They do not use the real `data/` folder and do not use internet access.

---

## 12. Suggested GitHub setup

### Files to commit

Commit source code and project files:

```text
backend/
frontend/
scripts/
tests/
requirements.txt
README.md
```

### Files usually not committed

Do not commit large local datasets or generated model files unless your teacher specifically requires them.

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/
.env

# Local data
data/raw/
data/*.parquet

# Generated models
models/*.joblib
models/model_metrics.json

# Test/cache files
.pytest_cache/
.coverage

# OS/editor files
.DS_Store
.vscode/
.idea/
```

If your class requires a fully runnable project without downloading data, ask whether generated Parquet files and model files should be included. In most GitHub projects, large raw datasets are not committed.

---

## 13. Git commands

Initialize Git:

```bash
git init
```

Check status:

```bash
git status
```

Add files:

```bash
git add backend frontend scripts tests requirements.txt README.md
```

Commit:

```bash
git commit -m "Initial InvestIQ project"
```

Add remote repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/InvestIQ.git
```

Push:

```bash
git branch -M main
git push -u origin main
```

---

## 14. Troubleshooting

### `Dataset not found. Run python scripts/build_dataset.py first.`

The backend cannot find:

```text
data/sp500_complete.parquet
data/sp500_prices.parquet
```

Run:

```bash
python scripts/build_dataset.py
```

### `Ticker 'XYZ' was not found`

The ticker is not available in the prepared local dataset.

Use autocomplete or check available tickers:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_parquet("data/sp500_complete.parquet")
print(sorted(df["ticker"].dropna().str.upper().unique())[:100])
PY
```

### `ModuleNotFoundError: No module named 'backend'`

Make sure you are running commands from the project root:

```bash
cd InvestIQ
```

Also make sure this file exists:

```text
backend/__init__.py
```

Create it if needed:

```bash
touch backend/__init__.py
```

### Plotly charts do not show

Make sure the browser can load Plotly from the CDN in `analysis.html`.

For a fully offline frontend, download Plotly locally and replace the CDN script with a local script path.

### Pydantic `field_validator` error

Upgrade FastAPI and Pydantic:

```bash
pip install --upgrade fastapi pydantic
```

---

## 15. Educational disclaimer

InvestIQ is an educational class project.

It is not investment advice.

The analysis uses historical local data and simplified financial rules. The ML valuation model learns from rule-based labels created from the local dataset. The backtest is illustrative and should not be interpreted as a real trading system.
