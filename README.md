# RAG Chatbot Project (OpenRouter) — Assignment-ready
This RAG project is configured to use **OpenRouter** for embeddings and chat completions. **For security, your API key is NOT included in the code** — set it as an environment variable before running.

## Important: do NOT commit your API key to GitHub
Use environment variables (instructions below) or a local `.env` file that is added to `.gitignore`.

## What this contains
- `ingest.py` — reads PDFs from `data/`, chunks text, creates embeddings and saves them to `store/`.
- `utils.py` — PDF reading, chunking, embedding call (uses OpenRouter via `openai` package), and a simple retriever using `sklearn`.
- `app.py` — Streamlit app that runs the chat UI and answers questions using retrieved context + OpenRouter chat completions.
- `requirements.txt`, `.gitignore`, and helper files.
- `data/PLACE_YOUR_PDFS_HERE.txt` — put your NCERT PDF(s) here.

## How to set your OpenRouter key (example)
**Do NOT** paste your key directly into files. Use environment variables:

macOS / Linux (bash/zsh):
```bash
export OPENROUTER_API_KEY="or-<your-key-here>"
```

Windows (PowerShell):
```powershell
setx OPENROUTER_API_KEY "or-<your-key-here>"
```

Restart the terminal or VS Code after `setx` on Windows to ensure the variable is available.

## Run steps
1. Create virtual env and activate
2. `pip install -r requirements.txt`
3. Put your NCERT Class 10 Maths PDF(s) in `data/`
4. `python ingest.py`  # creates `store/`
5. `streamlit run app.py`

## What to screenshot for assignment
- Terminal output after `python ingest.py` (shows store files saved)
- Streamlit UI with question + answer + retrieved passages
- VS Code file tree showing `app.py`, `utils.py`, and `README.md`
- GitHub repo page (after you push)

