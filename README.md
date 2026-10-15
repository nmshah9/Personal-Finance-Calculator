# Personal Finance AI Agent (LangChain + GPT-4o-mini + Streamlit)

An AI assistant where the **LLM chooses the tool** and **Python does the maths**: EMI, SIP future value,
goal-based SIP, 50-30-20 budgets and a safe calculator.

## Files
| File | Purpose |
|---|---|
| `Personal_Finance_Agent.ipynb` | Step-by-step notebook (build, test, evaluate) |
| `finance_tools.py` | Pure finance functions + LangChain `@tool` wrappers |
| `agent.py` | LLM, system prompt, `create_agent`, `ask()` helper with tool trace |
| `demo_queries.py` | The 5 required + 4 extra test queries with expected tools |
| `app.py` | Streamlit app: Calculators tab (enter amounts, no key needed) + AI chat |
| `run_demo.py` | CLI demo + transcript (`python run_demo.py --all`) |
| `tests/` | 45 offline tests (no API key needed) |

## Setup on Windows (CMD + VS Code)
```
cd personal-finance-agent
python -m venv financeenv
financeenv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env          (then edit .env and paste your real OpenAI key)
```
In VS Code choose the `financeenv` interpreter (Ctrl+Shift+P -> Python: Select Interpreter) and the same kernel in the notebook.

## Run
```
pytest -q                       # offline tests, no key needed -> 45 passed
python run_demo.py --all        # real LLM, prints tool choices, saves demo_transcript.md
streamlit run app.py            # web app at http://localhost:8501
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (`.env` and `secrets.toml` are git-ignored - never commit keys).
2. share.streamlit.io -> New app -> pick repo, branch, main file `app.py`.
3. App settings -> Secrets -> add `OPENAI_API_KEY = "sk-..."` (or let each visitor type a key in the sidebar).
4. Deploy; copy the public URL for your submission.

## Troubleshooting
* `OpenAI API key not found` - create `.env` or type the key in the sidebar.
* `ImportError: create_agent` - you have LangChain 0.x; run `pip install -U "langchain>=1.0,<2.0"`.
* Rupee sign garbled in a Windows console - run `chcp 65001` first (the notebook and app are unaffected).
