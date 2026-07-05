# GenAI Flask App — Structured JSON Outputs from Multiple LLMs on IBM watsonx.ai

Guided project from the IBM Skills Network lab **"Build Your First GenAI Application The Right Way"**, completed and extended for my AI/ML engineering portfolio.

A Flask web application that sends customer messages to a choice of three foundation models — **Llama 4 Maverick**, **IBM Granite 4**, or **Mistral Small 3.1** — on **watsonx.ai**, and returns **validated, structured JSON** (summary, sentiment score, suggested response, and recommended next step) via LangChain's `JsonOutputParser` with a Pydantic schema.

## Architecture

```
Browser (index.html + script.js)
        │  POST /generate {message, model}
        ▼
Flask (app.py)
        │
        ▼
model.py — per-model LCEL chain:
   PromptTemplate (model-specific special tokens)
        | ChatWatsonx (Llama / Granite / Mistral)
        | JsonOutputParser (Pydantic-validated AIResponse)
        │
        ▼
{"summary", "sentiment", "response", "next_step", "duration"}
```

## What's inside

| File | Purpose |
|---|---|
| `app.py` | Flask app: `/` frontend, `/generate` API with input validation and error handling |
| `model.py` | AI layer: 3 models, per-family special-token templates, Pydantic schema + JSON parser, LCEL chains |
| `config.py` | Centralized parameters, credentials (env-var aware), model IDs |
| `capital.py` | Standalone first-call demo with the raw `ibm-watsonx-ai` library, showing bare vs. special-token prompting |
| `llm_test.py` | Sanity check calling all three models with one prompt |
| `templates/index.html` | Chat UI (lab-provided markup) |
| `static/script.js`, `static/styles.css` | Frontend logic and styling — my own implementations against the lab HTML's element structure |
| `tests/test_offline.py` | 10-test offline suite (fake chat model + Flask test client) — **runs with zero credentials** |

## Key concepts demonstrated

- **Structured outputs**: a Pydantic `AIResponse` schema drives `JsonOutputParser.get_format_instructions()`, which injects the JSON contract into every prompt — turning free-text LLMs into a predictable API.
- **Model-specific prompt formatting**: Llama (`<|begin_of_text|>` / header roles), Mistral (`[INST]`), and Granite each get their own template — special tokens are what make model behavior controllable.
- **Lab exercise completed**: the schema is extended with a `next_step` field recommending the support representative's next action, with the system prompt updated to match.
- **Multi-model comparison**: identical requests routed to three providers for side-by-side evaluation of quality, style, and latency (response time is measured and returned).
- **Credential-free testing**: the offline suite drives the real LCEL chains through all three templates with a `FakeListChatModel` and exercises every Flask endpoint (200/400/500 paths) via the test client.

## Setup and run

```bash
git clone https://github.com/pumpuni07/genai-flask-app.git
cd genai-flask-app
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1) Verify everything offline (no credentials needed):
python tests/test_offline.py

# 2) Run against watsonx.ai:
export WATSONX_APIKEY="your-ibm-cloud-api-key"        # not needed inside Skills Network labs
export WATSONX_PROJECT_ID="your-watsonx-project-id"   # defaults to 'skills-network'
python app.py
# open http://127.0.0.1:5000
```

Or test the API directly:

```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"message": "I was charged twice this month!", "model": "granite"}'
```

**Note:** watsonx model catalogs evolve. Model IDs in `config.py` are from the lab's June 2026 catalog table — verify current availability in the [watsonx.ai foundation model catalog](https://www.ibm.com/products/watsonx-ai/foundation-models) and substitute if needed.

## Tech stack

Python · Flask · LangChain (LCEL, JsonOutputParser) · Pydantic · langchain-ibm · IBM watsonx.ai (Llama 4, Granite 4, Mistral Small 3.1) · Vanilla JS frontend

## Acknowledgments

Based on the IBM Skills Network guided project *"Build Your First GenAI Application The Right Way"*. Lab structure, `index.html` markup, and exercise design by the course authors. The CSS/JavaScript frontend assets, the completed exercise, the test suite, and configuration hardening are my own work (the lab's original gist-hosted CSS/JS are not redistributed here).

## Author

**Jack Pumpuni Frimpong-Manso**
AI/ML Engineer & Research Scientist | DAAD Scholar
GitHub: [pumpuni07](https://github.com/pumpuni07) · Portfolio: [jackpumpunifrimpongmanso.base44.app](https://jackpumpunifrimpongmanso.base44.app)
