# InsureAssist AI

A scaffold for an AI-powered insurance assistant that handles lead intake, quote generation, and MCP-based tool integrations.

## Project structure

- `app/` contains the application packages.
- `tests/` contains the automated tests.
- `data/` contains seed examples and sample data.
- `docker-compose.yml` provides a local run configuration.

## Getting started

1. Create a virtual environment.
2. Install dependencies with `pip install -e .`.
3. Start the API with `uvicorn app.api.main:app --reload`.
4. Visit `http://localhost:8000/docs` for the FastAPI docs.

## Environment

Copy `.env.example` to `.env` and update the values as needed.
