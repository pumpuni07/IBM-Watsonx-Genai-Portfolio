# LangChain Agent with Basic Tools — ReAct Agent on IBM watsonx.ai

Exercise 7 from the IBM Skills Network course **"Build Smarter AI Apps: Empower LLMs with LangChain"**, completed and extended for my AI/ML engineering portfolio.

A **ReAct agent** (Reasoning + Acting) that autonomously decides which tool to use — following the Thought → Action → Observation → Final Answer loop — built with LangChain's classic agent API and IBM Granite models on watsonx.ai.

## What's inside

| File | Purpose |
|---|---|
| `langchain_agent_basic_tools.ipynb` | Full walkthrough notebook — tools, ReAct prompt, agent, executor, tests |
| `solution.py` | The same solution as a standalone executable script |
| `test_offline.py` | Offline test suite: 10 tool unit tests + 4 full agent-loop tests using a scripted `FakeListLLM` — **runs without any watsonx credentials** |
| `requirements.txt` | Pinned versions (see version note below) |

## Engineering highlights

- **Security over convenience**: the exercise hint suggests `eval()` for the calculator — this solution instead uses a **whitelisted AST evaluator**, because tool inputs come from an LLM and `eval()` on LLM output is an injection vector. Attempts like `__import__('os')` are rejected (covered by a test).
- **Robust agent configuration**: `stop_sequences=["\nObservation"]` prevents the model from hallucinating tool outputs; `handle_parsing_errors=True` recovers from malformed generations; `max_iterations=5` guarantees termination.
- **Testable without credentials**: `test_offline.py` drives the complete AgentExecutor loop with a scripted fake LLM speaking the ReAct protocol — the agent machinery is verified independently of model access.
- **Agent-friendly error messages**: tools return descriptive errors as observations, enabling the agent to self-correct.

## Version note (important)

This exercise uses `create_react_agent` and `AgentExecutor` from `langchain.agents` — the classic agent API of **LangChain 0.3.x**. These were **removed in LangChain 1.x** (agents moved to a new `create_agent` API and LangGraph). The pinned `requirements.txt` is therefore not optional. Verified working against `langchain==0.3.27` / `langchain-ibm==0.3.15`.

## Setup and run

```bash
git clone https://github.com/pumpuni07/ibm-watsonx-genai-portfolio.git
cd ibm-watsonx-genai-portfolio/langchain-agent-basic-tools
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1) Verify everything offline (no credentials needed):
python test_offline.py

# 2) Run against watsonx.ai:
export WATSONX_APIKEY="your-ibm-cloud-api-key"        # not needed inside Skills Network labs
export WATSONX_PROJECT_ID="your-watsonx-project-id"   # defaults to 'skills-network'
python solution.py
```

**Note:** watsonx model catalogs evolve. If `ibm/granite-3-3-8b-instruct` is deprecated, substitute a current instruct model ID from the [watsonx.ai foundation model catalog](https://www.ibm.com/products/watsonx-ai/foundation-models).

## Tech stack

Python · LangChain 0.3 (ReAct agents, AgentExecutor) · langchain-ibm · IBM watsonx.ai · AST-based safe evaluation · Jupyter

## Acknowledgments

Exercise design and starter code from the IBM Skills Network course *"Build Smarter AI Apps: Empower LLMs with LangChain"*. All solution code, the safe-evaluator design, and the offline test harness are my own work.

## Author

**Jack Pumpuni Frimpong-Manso**
AI/ML Engineer & Research Scientist | DAAD Scholar
GitHub: [pumpuni07](https://github.com/pumpuni07) · Portfolio: [jackpumpunifrimpongmanso.base44.app](https://jackpumpunifrimpongmanso.base44.app)
