# RAG PDF QA Bot — LangChain + IBM watsonx.ai + Chroma + Gradio

Guided project from IBM Skills Network: **"Construct a QA Bot that Leverages LangChain and LLMs to Answer Questions from Loaded Documents"** — completed with an offline test suite added, for my AI/ML engineering portfolio.

Upload a PDF, ask a question, get an answer grounded in the document. A complete Retrieval-Augmented Generation pipeline:

```
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter (1000/50)
    -> WatsonxEmbeddings (ibm/slate-125m-english-rtrvr-v2) -> Chroma vector store
    -> retriever -> RetrievalQA (Mistral Medium on watsonx.ai) -> Gradio UI
```

## What's inside

| File | Purpose |
|---|---|
| `qabot.py` | The complete bot: loader, splitter, embeddings, vector DB, retriever, QA chain, Gradio interface |
| `sample_document.pdf` | Small sample PDF for demos and tests |
| `tests/test_offline.py` | 5-test offline suite — **zero credentials needed** |
| `requirements.txt` | Lab-pinned, verified-compatible versions |

## What the tests actually verify

The offline suite runs the REAL pipeline — real PDF parsing, real chunking, a real Chroma vector store, a real retriever, and a real RetrievalQA chain — patching only the two watsonx factories (`FakeListLLM` + `DeterministicFakeEmbedding`). End-to-end: a question about the sample PDF returns the expected grounded answer.

## Setup and run

```bash
git clone https://github.com/pumpuni07/ibm-watsonx-genai-portfolio.git
cd ibm-watsonx-genai-portfolio/rag-pdf-qa-bot
python3 -m venv my_env && source my_env/bin/activate
pip install -r requirements.txt

# 1) Verify offline (no credentials):
python tests/test_offline.py

# 2) Run against watsonx.ai:
export WATSONX_APIKEY="your-ibm-cloud-api-key"        # not needed inside Skills Network labs
export WATSONX_PROJECT_ID="your-watsonx-project-id"   # defaults to 'skills-network'
python qabot.py
# open http://127.0.0.1:7860 — upload a PDF, ask a question
```

**Notes:** best with smaller PDFs (in-memory Chroma). Verify `mistralai/mistral-medium-2505` and the Slate embedding model are still in the [watsonx catalog](https://www.ibm.com/products/watsonx-ai/foundation-models). One deviation from the lab: `launch()` is wrapped in `if __name__ == "__main__":` for importability in tests, and `share=False` by default.

## Tech stack

Python · LangChain (RetrievalQA) · Chroma · pypdf · langchain-ibm (WatsonxLLM, WatsonxEmbeddings) · IBM watsonx.ai · Gradio

## Acknowledgments

Based on the IBM Skills Network guided project above; lab design by the course authors. The test suite, sample document, and configuration hardening are my own work.

## Author

**Jack Pumpuni Frimpong-Manso** — AI/ML Engineer & Research Scientist | DAAD Scholar
GitHub: [pumpuni07](https://github.com/pumpuni07) · Portfolio: [jackpumpunifrimpongmanso.base44.app](https://jackpumpunifrimpongmanso.base44.app)
