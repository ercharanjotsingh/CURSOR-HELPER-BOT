# CURSOR-HELPER-BOT

A Python RAG chatbot that answers questions from the Markdown knowledge base in `data/`.

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy environment placeholders and add your own keys (never commit `.env`):

   ```powershell
   copy .env.example .env
   ```

## Status

Project scaffold only. Ingestion, retrieval, and chat are not implemented yet.
The knowledge base lives in `data/cursor_helper_kb.md`.
