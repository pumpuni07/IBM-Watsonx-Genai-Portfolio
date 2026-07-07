"""Offline test suite — no watsonx credentials required.

Verifies on the lab's pinned stack (gradio 4.44.0, langchain 0.2.11,
langchain-ibm 0.1.11):
  1. Core functions: add_numbers, combine_sentences, sentence_builder.
  2. Every Gradio interface constructs without launching.
  3. llm_chat.generate_response works end-to-end with a mocked WatsonxLLM.
  4. The exercise fix is in place (MAX_NEW_TOKENS raised in llm_chat).

Run from the project root:  python tests/test_offline.py
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Inject a fake WatsonxLLM BEFORE importing simple_llm / llm_chat, so no
# credentials or network access are needed at module import time.
# ---------------------------------------------------------------------------
class _FakeWatsonxLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, prompt):
        return f"[fake response to: {prompt}]"


fake_langchain_ibm = types.ModuleType("langchain_ibm")
fake_langchain_ibm.WatsonxLLM = _FakeWatsonxLLM
sys.modules["langchain_ibm"] = fake_langchain_ibm

import gradio as gr  # noqa: E402

import gradio_demo  # noqa: E402
import sentence_combiner  # noqa: E402
import common_input_types  # noqa: E402
import simple_llm  # noqa: E402
import llm_chat  # noqa: E402

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # noqa: E402


class TestFunctions(unittest.TestCase):
    def test_add_numbers(self):
        self.assertEqual(gradio_demo.add_numbers(3, 4), 7)
        self.assertEqual(gradio_demo.add_numbers(-2.5, 2.5), 0)

    def test_combine_sentences(self):
        self.assertEqual(
            sentence_combiner.combine_sentences("Hello world.", "Gradio is fun."),
            "Hello world. Gradio is fun.",
        )
        self.assertEqual(
            sentence_combiner.combine_sentences("  Trimmed.  ", ""),
            "Trimmed.",
        )

    def test_sentence_builder(self):
        result = common_input_types.sentence_builder(
            3, "Software Developer", ["Canada", "Japan"], "restaurant",
            ["coded", "fixed bugs"], True,
        )
        self.assertEqual(
            result,
            "The 3 Software Developers from Canada and Japan went to the "
            "restaurant where they coded and fixed bugs until the morning",
        )


class TestInterfacesConstruct(unittest.TestCase):
    def test_all_demos_are_gradio_interfaces(self):
        for module, attr in [
            (gradio_demo, "demo"),
            (sentence_combiner, "demo"),
            (common_input_types, "demo"),
            (llm_chat, "chat_application"),
        ]:
            obj = getattr(module, attr)
            self.assertIsInstance(obj, gr.Interface, f"{module.__name__}.{attr}")

    def test_common_input_types_has_examples(self):
        self.assertEqual(len(common_input_types.demo.examples), 4)


class TestLLMChat(unittest.TestCase):
    def test_generate_response_via_mocked_llm(self):
        out = llm_chat.generate_response("What is the capital of Canada?")
        self.assertIn("What is the capital of Canada?", out)

    def test_exercise_fix_max_new_tokens_raised(self):
        # llm_chat must use a higher token limit than simple_llm's 256
        self.assertEqual(simple_llm.parameters[GenParams.MAX_NEW_TOKENS], 256)
        self.assertGreater(llm_chat.parameters[GenParams.MAX_NEW_TOKENS], 256)

    def test_llm_configured_with_expected_settings(self):
        cfg = llm_chat.watsonx_llm.kwargs
        self.assertEqual(cfg["model_id"], "meta-llama/llama-3-2-11b-vision-instruct")
        self.assertEqual(cfg["url"], "https://us-south.ml.cloud.ibm.com")
        self.assertIn("project_id", cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
