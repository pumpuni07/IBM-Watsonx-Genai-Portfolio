"""Central configuration: model parameters, credentials, and model IDs.

Inside the IBM Skills Network Cloud IDE, no API key is needed
(project_id='skills-network' is handled automatically). Outside that
environment, export WATSONX_APIKEY and WATSONX_PROJECT_ID from your
IBM Cloud account.
"""

import os

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Model generation parameters
PARAMETERS = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 256,
}

# watsonx credentials
CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "project_id": os.environ.get("WATSONX_PROJECT_ID", "skills-network"),
    # None inside Skills Network (handled automatically); set the env var elsewhere
    "apikey": os.environ.get("WATSONX_APIKEY"),
}

# Model IDs (from the lab's watsonx catalog table, June 2026 — verify current
# availability at https://www.ibm.com/products/watsonx-ai/foundation-models)
LLAMA_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
GRANITE_MODEL_ID = "ibm/granite-4-h-small"
MISTRAL_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"
