# Gradio LLM Chatbot — Web Interfaces for watsonx.ai Models

Guided project from the IBM Skills Network lab **"Set Up a Simple Gradio Interface to Interact with Your Models"**, completed with both lab exercises solved and an offline test suite added, for my AI/ML engineering portfolio.

Builds from a first Gradio app up to a working web chatbot backed by **Llama 3.2** (or Mistral) on **IBM watsonx.ai** via LangChain's `WatsonxLLM` wrapper.

## What's inside

| File | Purpose |
|---|---|
| `gradio_demo.py` | First Gradio app: sum calculator with `gr.Number` inputs/outputs |
| `sentence_combiner.py` | **Lab exercise 1 solved**: app combining two input sentences |
| `common_input_types.py` | Tour of Gradio inputs — Slider, Dropdown, CheckboxGroup, Radio, multiselect, Checkbox — with an examples table |
| `simple_llm.py` | Terminal Q&A calling watsonx.ai through `WatsonxLLM` |
| `llm_chat.py` | Gradio web chatbot for the LLM — **lab exercise 2 solved**: truncated responses fixed by raising `MAX_NEW_TOKENS` (256 → 1024) |
| `tests/test_offline.py` | 8-test offline suite — **runs with zero credentials** |
| `requirements.txt` | The lab's pinned, verified-compatible versions |

## What the tests actually verify

- All app functions produce correct output (calculator, sentence combiner, sentence builder)
- Every Gradio `Interface` constructs successfully on the pinned stack
- The chatbot's `generate_response` flow works end-to-end with a mocked `WatsonxLLM`
- The exercise fix is present (`llm_chat` token limit > `simple_llm`'s 256)
- Additionally verified during development: a real Gradio server was launched, called over HTTP via `gradio_client`, returned the correct result, and shut down cleanly

## Setup and run

The lab targets Python 3.11; the suite was verified on Python 3.12 as well.

```bash
git clone https://github.com/pumpuni07/ibm-watsonx-genai-portfolio.git
cd ibm-watsonx-genai-portfolio/gradio-watsonx-llm-chatbot
python3 -m venv my_env && source my_env/bin/activate
pip install -r requirements.txt

# 1) Verify everything offline (no credentials needed):
python tests/test_offline.py

# 2) Run the apps:
python gradio_demo.py           # sum calculator  -> http://127.0.0.1:7860
python sentence_combiner.py     # exercise app    -> http://127.0.0.1:7860
python common_input_types.py    # input types tour -> http://127.0.0.1:7860
python llm_chat.py              # LLM chatbot     -> http://127.0.0.1:7860
```

For the LLM scripts (`simple_llm.py`, `llm_chat.py`):

- **Inside the IBM Skills Network Cloud IDE**, they run as-is (`project_id="skills-network"`, no API key).
- **Elsewhere**, export credentials from your IBM Cloud account first:

```bash
export WATSONX_APIKEY="your-ibm-cloud-api-key"
export WATSONX_PROJECT_ID="your-watsonx-project-id"
```

**Notes:** watsonx model catalogs evolve — if `meta-llama/llama-3-2-11b-vision-instruct` is deprecated, substitute a current model ID from the [watsonx.ai catalog](https://www.ibm.com/products/watsonx-ai/foundation-models). One small deviation from the lab: each `demo.launch(...)` is wrapped in `if __name__ == "__main__":` so the modules are importable for testing — behavior when running the scripts directly is unchanged.

## Tech stack

Python · Gradio 4 · LangChain · langchain-ibm (`WatsonxLLM`) · IBM watsonx.ai (Llama 3.2 / Mistral)

## Acknowledgments

Based on the IBM Skills Network lab *"Set Up a Simple Gradio Interface to Interact with Your Models"*. Lab structure and demo code by the course authors; the exercise solutions, testability refactor, and offline test suite are my own work.

## Author

**Jack Pumpuni Frimpong-Manso**
AI/ML Engineer & Research Scientist | DAAD Scholar
GitHub: [pumpuni07](https://github.com/pumpuni07) · Portfolio: [jackpumpunifrimpongmanso.base44.app](https://jackpumpunifrimpongmanso.base44.app)
