"""Gradio chatbot backed by a watsonx.ai LLM.

Lab exercise applied: responses were occasionally cut off because
MAX_NEW_TOKENS limited output length — the fix is raising the token limit
(here 256 -> 1024), which lets the model complete longer answers.
"""

import os

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM
import gradio as gr

# Model and project settings (swap the comment to try Mistral instead)
# model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503'  # Mistral model
model_id = "meta-llama/llama-3-2-11b-vision-instruct"  # Llama model

# Generation parameters
# Exercise fix: MAX_NEW_TOKENS raised from 256 to 1024 so longer answers
# are not truncated mid-sentence.
parameters = {
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0.5,
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


# Function to generate a response from the model
def generate_response(prompt_txt):
    generated_response = watsonx_llm.invoke(prompt_txt)
    return generated_response


# Create the Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer.",
)

# Launch the app
if __name__ == "__main__":
    chat_application.launch(server_name="127.0.0.1", server_port=7860)
