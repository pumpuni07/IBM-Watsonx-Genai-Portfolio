"""Central configuration for the Icebreaker Bot.

Note: the original lab ships these templates in a starter tarball; the
templates below are my own equivalents serving the same purpose.
"""

import os

# --- watsonx.ai --------------------------------------------------------
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "skills-network")
WATSONX_APIKEY = os.environ.get("WATSONX_APIKEY")

LLM_MODEL_ID = "ibm/granite-4-h-small"
EMBEDDING_MODEL_ID = "ibm/slate-125m-english-rtrvr-v2"

# --- Generation parameters --------------------------------------------
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 500
MIN_NEW_TOKENS = 1
TOP_K = 50
TOP_P = 0.9

# --- RAG parameters ----------------------------------------------------
CHUNK_SIZE = 500
SIMILARITY_TOP_K = 5

# --- Data extraction ---------------------------------------------------
# ProxyCurl was discontinued (Feb 2025); mock data is the working path.
PROXYCURL_API_KEY = os.environ.get("PROXYCURL_API_KEY", "")
MOCK_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_data", "mock_profile.json")

# --- Prompt templates (own implementations) ----------------------------
INITIAL_FACTS_TEMPLATE = (
    "Context information about a person's LinkedIn profile is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using ONLY the context above and no prior knowledge, provide three "
    "interesting, specific facts about this person's career or education. "
    "Number them 1 to 3 and keep each fact to one or two sentences.\n"
    "Query: {query_str}\n"
    "Answer: "
)

USER_QUESTION_TEMPLATE = (
    "Context information about a person's LinkedIn profile is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using ONLY the context above and no prior knowledge, answer the query. "
    "If the answer is not in the context, say \"I don't know\".\n"
    "Query: {query_str}\n"
    "Answer: "
)
