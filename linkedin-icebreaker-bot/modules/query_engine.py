"""Query engine: fact generation and question answering over the profile index."""

import logging
from typing import Any

from llama_index.core import PromptTemplate, VectorStoreIndex

import config
from modules.llm_interface import create_watsonx_llm

logger = logging.getLogger(__name__)


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generates interesting facts about the person's career or education."""
    try:
        watsonx_llm = create_watsonx_llm(
            temperature=0.0,
            max_new_tokens=500,
            decoding_method="sample",
        )
        facts_prompt = PromptTemplate(template=config.INITIAL_FACTS_TEMPLATE)
        query_engine = index.as_query_engine(
            streaming=False,
            similarity_top_k=config.SIMILARITY_TOP_K,
            llm=watsonx_llm,
            text_qa_template=facts_prompt,
        )
        query = "Provide three interesting facts about this person's career or education."
        response = query_engine.query(query)
        return response.response
    except Exception as e:
        logger.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."


def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """Answers the user's question using the vector database and the LLM."""
    try:
        watsonx_llm = create_watsonx_llm(
            temperature=0.0,
            max_new_tokens=250,
            decoding_method="greedy",
        )
        question_prompt = PromptTemplate(template=config.USER_QUESTION_TEMPLATE)
        query_engine = index.as_query_engine(
            streaming=False,
            similarity_top_k=config.SIMILARITY_TOP_K,
            llm=watsonx_llm,
            text_qa_template=question_prompt,
        )
        return query_engine.query(user_query)
    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer."
