"""AI model integration layer.

Three watsonx.ai chat models (Llama, Granite, Mistral), each with its own
special-token prompt template, chained through LangChain's LCEL pipe operator
into a JsonOutputParser for structured, validated JSON output.

Exercise completed: the AIResponse schema includes a `next_step` field
recommending the support representative's next action.
"""

from langchain_ibm import ChatWatsonx
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from config import PARAMETERS, CREDENTIALS, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID


# ---------------------------------------------------------------------------
# Structured output schema (Pydantic)
# ---------------------------------------------------------------------------
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")
    # Exercise: recommend the next step for the support representative
    next_step: str = Field(
        description="Recommended next step the support representative should take to resolve this issue"
    )


# JSON output parser — validates and parses model output into the schema above
json_parser = JsonOutputParser(pydantic_object=AIResponse)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
def initialize_model(model_id):
    """Create a ChatWatsonx instance for the given model ID."""
    return ChatWatsonx(
        model_id=model_id,
        url=CREDENTIALS["url"],
        project_id=CREDENTIALS["project_id"],
        apikey=CREDENTIALS["apikey"],
        params=PARAMETERS,
    )


# Initialize models
llama_llm = initialize_model(LLAMA_MODEL_ID)
granite_llm = initialize_model(GRANITE_MODEL_ID)
mistral_llm = initialize_model(MISTRAL_MODEL_ID)


# ---------------------------------------------------------------------------
# Model-specific prompt templates (special tokens per model family)
# ---------------------------------------------------------------------------
llama_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}\n{format_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "format_prompt", "user_prompt"],
)

granite_template = PromptTemplate(
    template="System: {system_prompt}\n{format_prompt}\nHuman: {user_prompt}\nAI:",
    input_variables=["system_prompt", "format_prompt", "user_prompt"],
)

mistral_template = PromptTemplate(
    template="<s>[INST]{system_prompt}\n{format_prompt}\n{user_prompt}[/INST]",
    input_variables=["system_prompt", "format_prompt", "user_prompt"],
)


# ---------------------------------------------------------------------------
# Chain: template | model | json_parser  (LCEL pipe operator)
# ---------------------------------------------------------------------------
def get_ai_response(model, template, system_prompt, user_prompt):
    """Format the prompt, call the model, and parse the output into JSON.

    json_parser.get_format_instructions() injects the JSON schema derived
    from the AIResponse Pydantic model into the prompt, instructing the LLM
    to respond in exactly that structure.
    """
    chain = template | model | json_parser
    return chain.invoke({
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "format_prompt": json_parser.get_format_instructions(),
    })


# Model-specific response functions
def llama_response(system_prompt, user_prompt):
    return get_ai_response(llama_llm, llama_template, system_prompt, user_prompt)


def granite_response(system_prompt, user_prompt):
    return get_ai_response(granite_llm, granite_template, system_prompt, user_prompt)


def mistral_response(system_prompt, user_prompt):
    return get_ai_response(mistral_llm, mistral_template, system_prompt, user_prompt)
