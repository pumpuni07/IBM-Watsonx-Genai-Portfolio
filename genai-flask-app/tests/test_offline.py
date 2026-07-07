"""Offline test suite — no watsonx credentials required.

Verifies:
  1. The full LCEL chain (template | model | json_parser) parses structured
     JSON through each of the three model-specific templates, using a fake
     chat model.
  2. All three prompt templates have the required input variables and
     render with special tokens intact.
  3. The AIResponse schema includes the exercise's next_step field.
  4. Flask endpoints: GET / renders, POST /generate validates input and
     returns structured JSON (model functions mocked).

Run from the project root:  python -m tests.test_offline
"""

import json
import os
import sys
import types
import unittest
from unittest.mock import patch

# Ensure the project root is importable regardless of where tests run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Inject a fake ChatWatsonx BEFORE importing model.py, so no credentials or
# network access are needed. The fake is a real LangChain chat model, so the
# LCEL pipe composition is exercised for real.
# ---------------------------------------------------------------------------
from langchain_core.language_models.fake_chat_models import FakeListChatModel

FAKE_JSON = json.dumps({
    "summary": "Customer reports a double charge on their subscription.",
    "sentiment": 25,
    "response": "I'm sorry about the double charge. We'll investigate and refund the duplicate immediately.",
    "next_step": "Escalate to the billing team to verify the duplicate transaction and issue a refund.",
})


def _fake_chat_watsonx(**kwargs):
    return FakeListChatModel(responses=[FAKE_JSON] * 50)


fake_langchain_ibm = types.ModuleType("langchain_ibm")
fake_langchain_ibm.ChatWatsonx = _fake_chat_watsonx
sys.modules["langchain_ibm"] = fake_langchain_ibm

import model  # noqa: E402  (imports the fake ChatWatsonx above)
import app as flask_app_module  # noqa: E402


class TestSchema(unittest.TestCase):
    def test_airesponse_has_all_fields_including_exercise(self):
        fields = set(model.AIResponse.model_fields.keys())
        self.assertEqual(fields, {"summary", "sentiment", "response", "next_step"})

    def test_format_instructions_mention_next_step(self):
        instructions = model.json_parser.get_format_instructions()
        self.assertIn("next_step", instructions)


class TestTemplates(unittest.TestCase):
    REQUIRED = {"system_prompt", "format_prompt", "user_prompt"}

    def test_all_templates_have_required_variables(self):
        for name, tpl in [
            ("llama", model.llama_template),
            ("granite", model.granite_template),
            ("mistral", model.mistral_template),
        ]:
            self.assertEqual(set(tpl.input_variables), self.REQUIRED, name)

    def test_special_tokens_render(self):
        rendered = model.llama_template.format(
            system_prompt="S", format_prompt="F", user_prompt="U"
        )
        for token in ["<|begin_of_text|>", "<|start_header_id|>", "<|eot_id|>"]:
            self.assertIn(token, rendered)
        rendered = model.mistral_template.format(
            system_prompt="S", format_prompt="F", user_prompt="U"
        )
        self.assertIn("[INST]", rendered)
        self.assertIn("[/INST]", rendered)


class TestChain(unittest.TestCase):
    def test_all_three_response_functions_return_parsed_dicts(self):
        for fn in [model.llama_response, model.granite_response, model.mistral_response]:
            result = fn("system", "My subscription was charged twice!")
            self.assertIsInstance(result, dict)
            self.assertEqual(
                set(result.keys()), {"summary", "sentiment", "response", "next_step"}
            )
            self.assertIsInstance(result["sentiment"], int)


class TestFlaskEndpoints(unittest.TestCase):
    def setUp(self):
        flask_app_module.app.config["TESTING"] = True
        self.client = flask_app_module.app.test_client()

    def test_index_renders(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"AI Assistant", resp.data)

    def test_generate_missing_fields_returns_400(self):
        resp = self.client.post("/generate", json={"message": "hi"})
        self.assertEqual(resp.status_code, 400)
        resp = self.client.post("/generate", json={"model": "llama"})
        self.assertEqual(resp.status_code, 400)

    def test_generate_invalid_model_returns_400(self):
        resp = self.client.post(
            "/generate", json={"message": "hi", "model": "gpt-99"}
        )
        self.assertEqual(resp.status_code, 400)

    def test_generate_returns_structured_json_with_duration(self):
        for model_name in ["llama", "granite", "mistral"]:
            resp = self.client.post(
                "/generate",
                json={"message": "I was charged twice", "model": model_name},
            )
            self.assertEqual(resp.status_code, 200, model_name)
            data = resp.get_json()
            self.assertEqual(
                set(data.keys()),
                {"summary", "sentiment", "response", "next_step", "duration"},
                model_name,
            )
            self.assertIsInstance(data["duration"], float)

    def test_generate_model_error_returns_500(self):
        with patch.object(
            flask_app_module, "llama_response", side_effect=RuntimeError("boom")
        ):
            resp = self.client.post(
                "/generate", json={"message": "hi", "model": "llama"}
            )
            self.assertEqual(resp.status_code, 500)
            self.assertIn("boom", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
