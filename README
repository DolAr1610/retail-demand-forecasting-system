# Retail Demand Forecasting System

An AI-powered retail demand forecasting system for predicting product demand, comparing machine learning models, and supporting procurement decisions through an interactive web interface and AI assistant.

## Overview

This project helps retail teams forecast demand, analyze model performance, and generate replenishment recommendations based on historical sales data, current stock, safety stock, and lead time.

The system combines:

- machine learning demand forecasting
- model comparison and evaluation
- FastAPI backend
- static web frontend
- AI procurement copilot with tool-calling logic

## Key Features

- Demand forecasting by store, item, date, and model
- Global model metrics and comparison
- Interactive frontend for prediction and analytics
- AI assistant for forecast explanation and order recommendations
- Artifact-based model loading for trained ML outputs

## Architecture

```text
User / Browser
      |
      v
Frontend UI
      |
      v
FastAPI Backend
  |        |         |
  |        |         +--> AI Agent
  |        +------------> Metrics Service
  +---------------------> Prediction Service
                              |
                              v
                    Trained Model Artifacts
```

## Project Structure

```text
retail-demand-forecasting-system/
└── app/
    ├── backend/
    │   ├── api/          # FastAPI routes
    │   ├── models/       # Pydantic schemas
    │   ├── prompts/      # Agent prompts
    │   ├── services/     # Prediction, metrics, agent and LLM logic
    │   ├── utils/        # Helper utilities
    │   ├── main.py       # Application entry point
    │   └── settings.py   # Environment configuration
    ├── frontend/         # Static HTML/CSS/JS interface
    └── notebooks/        # Experiments and model training
```

## Tech Stack

**Backend:** FastAPI, Uvicorn, Pydantic, Pandas, NumPy  
**Machine Learning:** scikit-learn, XGBoost, LightGBM  
**Frontend:** HTML, CSS, JavaScript  
**AI Assistant:** LLM-based tool-calling agent  
**Artifacts:** trained forecasting models and prepared metadata

## Installation

Clone the repository:

```bash
git clone https://github.com/DolAr1610/retail-demand-forecasting-system.git
cd retail-demand-forecasting-system
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r app/backend/requirements.txt
```

## Configuration

Create a `.env` file in the project root if needed:

```env
FAVORITA_ARTIFACTS_DIR=data/artifacts/active
FAVORITA_LOG_LEVEL=INFO
FAVORITA_CORS_ALLOW_ORIGINS=*
FAVORITA_DEFAULT_STORE_NBR=45
```

Optional LLM settings:

```env
FAVORITA_OPENROUTER_API_KEY=your_api_key_here
FAVORITA_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
FAVORITA_OPENROUTER_MODEL=openrouter/free
```

If no LLM API key is provided, the assistant works in deterministic forecast mode.

## Running the Application

Start the backend from the project root:

```bash
uvicorn app.backend.main:app --reload
```

Open in browser:

```text
http://127.0.0.1:8000/ui
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Main Pages

- **Home** — project overview and model summary
- **Predict** — demand forecast lookup
- **Analytics** — model quality comparison
- **Agent** — AI assistant for forecasts and replenishment
- **About** — project purpose and workflow

## AI Agent Logic

The assistant does not generate forecast values manually. It uses backend tools to:

1. identify the requested store, item, date, and model
2. retrieve forecast data
3. explain prediction results
4. recommend replenishment quantities
5. return the answer in business-friendly language

## Use Cases

- retail demand forecasting
- procurement decision support
- inventory planning prototype
- ML model comparison demo
- academic thesis or capstone project

## Status

This project is an MVP focused on demonstrating how machine learning forecasts can be connected with a practical business interface and AI-based decision support.

## Author

Created by **Arsen Dolichnyi**.