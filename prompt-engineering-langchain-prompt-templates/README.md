# Master Prompt Engineering and LangChain PromptTemplates

Hands-on lab project from the **IBM RAG and Agentic AI Professional Certificate** (Coursera / IBM Skills Network), independently reconstructed and extended for my AI/ML engineering portfolio.

The notebook progresses from core prompt engineering techniques to composable LLM pipelines built with LangChain's modern **LCEL** (LangChain Expression Language) pattern, using **IBM watsonx.ai** foundation models (Granite instruct).

## Contents

| Section | Topic |
|---|---|
| 1 | Setup — watsonx.ai model initialization via `langchain-ibm` |
| 2 | Prompt engineering techniques — basic, zero-shot, one-shot, few-shot in-context learning |
| 3 | Advanced reasoning — chain-of-thought (CoT) prompting and self-consistency with majority voting |
| 4 | LangChain templates — `PromptTemplate`, `ChatPromptTemplate`, `FewShotPromptTemplate`, and LCEL chains with the pipe operator (`\|`) |
| 5 | Applications — grounded Q&A, JSON information extraction, tone-controlled translation, and a two-step classify-then-respond pipeline |
| 6 | Exercises with worked solutions |

## Key skills demonstrated

- **In-context learning**: teaching custom label schemes and output formats without fine-tuning
- **Chain-of-thought + self-consistency**: sampling multiple reasoning paths and aggregating by majority vote for reliability
- **LCEL composition**: `prompt | llm | StrOutputParser()` — modular, reusable chains that pipeline into one another
- **Grounded generation**: context-restricted Q&A with explicit refusal behavior (the conceptual core of RAG)
- **Prompt templating as engineering practice**: parameterized, versionable prompt components

## Setup

```bash
git clone https://github.com/pumpuni07/ibm-watsonx-genai-portfolio.git
cd ibm-watsonx-genai-portfolio/prompt-engineering-langchain-prompt-templates
pip install -r requirements.txt
```

The notebook targets IBM watsonx.ai:

- **Inside the IBM Skills Network lab environment**, it runs as-is (`project_id="skills-network"`, no API key needed).
- **In your own environment**, set credentials from your IBM Cloud account before launching:

```bash
export WATSONX_APIKEY="your-ibm-cloud-api-key"
export WATSONX_PROJECT_ID="your-watsonx-project-id"
```

Then run:

```bash
jupyter notebook prompt_engineering_langchain_prompt_templates.ipynb
```

**Note:** watsonx model catalogs evolve. If `ibm/granite-3-3-8b-instruct` is deprecated, substitute any current instruct model ID from the [watsonx.ai foundation model catalog](https://www.ibm.com/products/watsonx-ai/foundation-models).

## Tech stack

Python · LangChain (LCEL) · langchain-ibm · IBM watsonx.ai · Jupyter

## Acknowledgments

Independent reconstruction of the IBM Skills Network lab *"Master Prompt Engineering and LangChain PromptTemplates"*. Original lab authors: [Hailey Quach](https://www.haileyq.com/), Kang Wang, and Faranak Heidari — Data Scientists at IBM.

## Author

**Jack Pumpuni Frimpong-Manso**
AI/ML Engineer & Research Scientist | DAAD Scholar
GitHub: [pumpuni07](https://github.com/pumpuni07) · Portfolio: [jackpumpunifrimpongmanso.base44.app](https://jackpumpunifrimpongmanso.base44.app)
