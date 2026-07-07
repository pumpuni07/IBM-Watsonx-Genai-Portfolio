"""Terminal Q&A with a watsonx.ai LLM via LangChain's WatsonxLLM wrapper.

Inside the IBM Skills Network Cloud IDE, project_id='skills-network' works
without an API key. Elsewhere, export WATSONX_APIKEY and WATSONX_PROJECT_ID.
"""

import os

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM

# Specify the model (swap the comment to try Mistral instead)
# model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503'  # Mistral model
model_id = "meta-llama/llama-3-2-11b-vision-instruct"  # Llama model

# Generation parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,   # max tokens to generate
    GenParams.TEMPERATURE: 0.5,      # randomness / creativity
}

project_id = os.environ.get("WATSONX_PROJECT_ID", "skills-network")

# Wrap the model into a WatsonxLLM inference object
watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url="https://us-south.ml.cloud.ibm.com",
    project_id=project_id,
    apikey=os.environ.get("WATSONX_APIKEY"),
    params=parameters,
)

if __name__ == "__main__":
    # Get the query from user input and print the generated response
    query = input("Please enter your query: ")
    print(watsonx_llm.invoke(query))
