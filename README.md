# IBM watsonx.ai GenAI Engineering Portfolio

Six hands-on generative AI applications built on **IBM watsonx.ai**, spanning the core techniques of modern LLM engineering: prompt engineering, LCEL pipelines, structured outputs, agents, RAG with two different frameworks, and web frontends. Each project originates from an IBM Skills Network lab or Professional Certificate exercise, completed and extended with engineering work beyond the lab scope — and **every project ships a credential-free offline test suite**, so the pipelines are verifiable on any machine without watsonx access.

## Projects

| # | Project | What it demonstrates | Stack highlights |
|---|---|---|---|
| 1 | [`prompt-engineering-langchain-prompt-templates`](./prompt-engineering-langchain-prompt-templates) | Zero/one/few-shot prompting, chain-of-thought, self-consistency; LCEL chains for grounded Q&A, JSON extraction, translation | LangChain LCEL · PromptTemplate family · Granite |
| 2 | [`langchain-agent-basic-tools`](./langchain-agent-basic-tools) | ReAct agent with custom tools; **security-hardened**: whitelisted AST evaluator replaces unsafe `eval()` on LLM-controlled input | LangChain agents · AgentExecutor · injection tests |
| 3 | [`genai-flask-app`](./genai-flask-app) | Full-stack app returning Pydantic-validated JSON from three models (Llama 4 / Granite 4 / Mistral) with model-specific special-token prompts | Flask · JsonOutputParser · vanilla-JS frontend |
| 4 | [`gradio-watsonx-llm-chatbot`](./gradio-watsonx-llm-chatbot) | Gradio interfaces from first app to LLM chatbot; both lab exercises solved; server verified live over HTTP | Gradio 4 · WatsonxLLM |
| 5 | [`rag-pdf-qa-bot`](./rag-pdf-qa-bot) | RAG over uploaded PDFs: load → chunk → embed → retrieve → answer, grounded in the document | LangChain RetrievalQA · Chroma · Slate embeddings |
| 6 | [`linkedin-icebreaker-bot`](./linkedin-icebreaker-bot) | Modular RAG application with a second framework; CLI + session-managed web UI, runtime model switching | LlamaIndex · VectorStoreIndex · Gradio Blocks |

## Engineering practices demonstrated across the portfolio

- **Testing without credentials**: every project includes an offline suite driving the real pipeline (real chains, real vector stores, real agent loops) with faked model backends — 40+ tests total across the portfolio.
- **Security awareness**: LLM-controlled inputs are never passed to `eval()`; the agent project documents and tests the injection-safe alternative.
- **Structured outputs**: Pydantic schemas + parsers turn free-text LLMs into predictable APIs.
- **Version discipline**: dependency conflicts diagnosed and pinned (e.g., the classic LangChain agent API removed in 1.x), with verified-compatible `requirements.txt` per project.
- **Honest attribution**: each README states exactly what comes from the lab and what is my own work.

## Getting started

Each project is self-contained with its own `README.md`, `requirements.txt`, and tests:

```bash
git clone https://github.com/pumpuni07/ibm-watsonx-genai-portfolio.git
cd ibm-watsonx-genai-portfolio/<project-folder>
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python tests/test_offline.py   # verify with zero credentials
```

Running against live watsonx.ai requires `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` (not needed inside IBM Skills Network labs). Model IDs should be checked against the current [watsonx.ai catalog](https://www.ibm.com/products/watsonx-ai/foundation-models), as catalogs evolve.

## Author

**Jack Pumpuni Frimpong-Manso**
AI/ML Engineer & Research Scientist | DAAD Scholar
Python · PyTorch · Machine Learning · LLMs · RAG · MLOps
GitHub: [pumpuni07](https://github.com/pumpuni07) · Portfolio: [jackpumpunifrimpongmanso.base44.app](https://jackpumpunifrimpongmanso.base44.app)
