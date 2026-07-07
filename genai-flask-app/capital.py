"""First direct call to watsonx.ai using the ibm-watsonx-ai library.

Demonstrates the difference between a bare prompt and a prompt formatted
with Llama's special tokens (roles: system / user / assistant).
"""

import os

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    # No API key needed inside the Skills Network Cloud IDE.
    # Elsewhere: export WATSONX_APIKEY from your IBM Cloud account.
    api_key=os.environ.get("WATSONX_APIKEY"),
)

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
}

model = ModelInference(
    model_id="ibm/granite-4-h-small",
    params=params,
    credentials=credentials,
    project_id=os.environ.get("WATSONX_PROJECT_ID", "skills-network"),
)

# --- Bare prompt -----------------------------------------------------------
text = """
Only reply with the answer. What is the capital of Canada?
"""
print("Bare prompt:")
print(model.generate(text)["results"][0]["generated_text"])

# --- Prompt with Llama special tokens (swap model_id to a Llama model) ------
llama_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant who provides concise and accurate answers.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
What is the capital of Canada?<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
# To test: set model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
# above and run model.generate(llama_text) — special tokens give the model
# role structure, producing a clean, controlled answer.
