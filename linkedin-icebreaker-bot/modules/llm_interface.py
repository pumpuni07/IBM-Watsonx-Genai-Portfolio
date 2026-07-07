"""Interface to IBM watsonx.ai language and embedding models (LlamaIndex)."""

import logging

from llama_index.llms.ibm import WatsonxLLM
from llama_index.embeddings.ibm import WatsonxEmbeddings

import config

logger = logging.getLogger(__name__)


def create_watsonx_embedding() -> WatsonxEmbeddings:
    """Creates an IBM watsonx embedding model for vector representation."""
    watsonx_embedding = WatsonxEmbeddings(
        model_id=config.EMBEDDING_MODEL_ID,
        url=config.WATSONX_URL,
        project_id=config.WATSONX_PROJECT_ID,
        apikey=config.WATSONX_APIKEY,
        truncate_input_tokens=3,
    )
    logger.info(f"Created Watsonx Embedding model: {config.EMBEDDING_MODEL_ID}")
    return watsonx_embedding


def create_watsonx_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample",
) -> WatsonxLLM:
    """Creates an IBM watsonx LLM for generating responses."""
    additional_params = {
        "decoding_method": decoding_method,
        "min_new_tokens": config.MIN_NEW_TOKENS,
        "top_k": config.TOP_K,
        "top_p": config.TOP_P,
    }
    watsonx_llm = WatsonxLLM(
        model_id=config.LLM_MODEL_ID,
        url=config.WATSONX_URL,
        project_id=config.WATSONX_PROJECT_ID,
        apikey=config.WATSONX_APIKEY,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        additional_params=additional_params,
    )
    logger.info(f"Created Watsonx LLM model: {config.LLM_MODEL_ID}")
    return watsonx_llm


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use."""
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")
