# LinkedIn Icebreaker Bot — RAG with LlamaIndex + IBM Granite on watsonx.ai

Guided project from IBM Skills Network: **"Build an AI Icebreaker Bot with LlamaIndex & IBM Granite"** — reconstructed, completed, and test-covered for my AI/ML engineering portfolio.

Give it a LinkedIn profile (mock data included), and it builds a RAG pipeline over the profile, generates three personalized conversation starters, and answers follow-up questions — via both a CLI and a Gradio web UI.

## Architecture (RAG pipeline)

```
Profile JSON (mock file or ProxyCurl-style API)
  -> SentenceSplitter (chunk_size 500) -> nodes
  -> WatsonxEmbeddings (ibm/slate-125m-english-rtrvr-v2) -> VectorStoreIndex
  -> query engine (similarity_top_k=5, custom prompt templates)
  -> IBM Granite 4 (watsonx.ai) -> icebreaker facts / grounded answers
```

## What's inside

| File | Purpose |
|---|---|
| `config.py` | Central settings: models, RAG parameters, prompt templates |
| `modules/data_extraction.py` | Profile extraction: local mock JSON or ProxyCurl-style API, with data cleaning |
| `modules/data_processing.py` | Splitting, vector indexing, embedding verification |
| `modules/llm_interface.py` | watsonx LLM + embedding factories (LlamaIndex), runtime model switching |
| `modules/query_engine.py` | Fact generation + grounded Q&A with custom templates |
| `main.py` | CLI with argparse (`--mock`, `--url`, `--api-key`, `--model`) |
| `app.py` | Gradio Blocks web UI: process tab + session-based chat tab |
| `mock_data/mock_profile.json` | Fictional sample profile (ProxyCurl was discontinued in Feb 2025 — mock is the working path) |
| `tests/test_offline.py` | 8-test offline suite — **zero credentials needed** |

## What the tests actually verify

The REAL LlamaIndex pipeline runs — real `SentenceSplitter`, real `VectorStoreIndex`, real query engines — with only the two watsonx factories patched (LlamaIndex's built-in `MockLLM`/`MockEmbedding`): mock extraction + data cleaning, node splitting, index creation, embedding verification, fact generation, grounded Q&A, Gradio Blocks construction, and session handling.

## Setup and run

```bash
git clone https://github.com/pumpuni07/ibm-watsonx-genai-portfolio.git
cd ibm-watsonx-genai-portfolio/linkedin-icebreaker-bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1) Verify offline (no credentials):
python tests/test_offline.py

# 2) Run against watsonx.ai:
export WATSONX_APIKEY="your-ibm-cloud-api-key"        # not needed inside Skills Network labs
export WATSONX_PROJECT_ID="your-watsonx-project-id"   # defaults to 'skills-network'

python main.py --mock          # CLI
python app.py                  # web UI -> http://127.0.0.1:5000
```

## Disclosed deviations from the lab

The lab ships a starter tarball (not redistributable here), so parts not shown in the lab text are my own equivalents: the prompt templates in `config.py`, and the mock profile — a **fictional** profile shipped locally in the repo instead of the lab's remote mock URL. All implemented functions follow the lab's specifications. `launch()` calls are wrapped in `__main__` guards for testability.

## Tech stack

Python · LlamaIndex (VectorStoreIndex, query engines) · llama-index-llms-ibm / llama-index-embeddings-ibm · IBM watsonx.ai (Granite 4, Slate embeddings) · Gradio Blocks · RAG

## Acknowledgments

Based on the IBM Skills Network guided project above; lab design by the course authors. Prompt templates, mock data, test suite, and hardening are my own work.

## Author

**Jack Pumpuni Frimpong-Manso** — AI/ML Engineer & Research Scientist | DAAD Scholar
GitHub: [pumpuni07](https://github.com/pumpuni07) · Portfolio: [jackpumpunifrimpongmanso.base44.app](https://jackpumpunifrimpongmanso.base44.app)
