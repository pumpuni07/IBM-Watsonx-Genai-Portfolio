"""Sanity check: call all three models with the same prompt.

Note: after the JsonOutputParser was added to the chain in model.py, the
response functions return a parsed dict, not an AIMessage — so we print
dict fields here (the lab's earlier `.content` version no longer applies).
"""

import json

from model import llama_response, granite_response, mistral_response


def call_all_models(system_prompt, user_prompt):
    for name, fn in [
        ("Llama", llama_response),
        ("Granite", granite_response),
        ("Mistral", mistral_response),
    ]:
        result = fn(system_prompt, user_prompt)
        print(f"\n{name} Response:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    call_all_models(
        "You are a helpful assistant who provides concise and accurate answers",
        "What is the capital of Canada? Tell me a cool fact about it as well",
    )
