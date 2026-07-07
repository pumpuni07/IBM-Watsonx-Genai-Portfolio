"""Offline test suite for the Icebreaker Bot — no watsonx credentials required.

Runs the REAL LlamaIndex pipeline (SentenceSplitter, VectorStoreIndex,
retriever, query engine) using llama_index's built-in MockLLM and
MockEmbedding, patching only the two watsonx factory functions.

Run from the project root:  python tests/test_offline.py
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core import MockEmbedding
from llama_index.core.llms import MockLLM

import config
from modules import data_extraction, data_processing, query_engine

fake_llm = lambda **kw: MockLLM(max_tokens=64)
fake_emb = lambda: MockEmbedding(embed_dim=64)


class TestDataExtraction(unittest.TestCase):
    def test_mock_extraction_loads_and_cleans(self):
        data = data_extraction.extract_linkedin_profile("any-url", mock=True)
        self.assertEqual(data["full_name"], "Alex Morgan Tan")
        self.assertNotIn("certifications", data)  # cleaned (unwanted/empty)

    def test_api_mode_without_key_returns_empty(self):
        data = data_extraction.extract_linkedin_profile("any-url", api_key=None, mock=False)
        self.assertEqual(data, {})


class TestDataProcessing(unittest.TestCase):
    def test_split_profile_data_creates_nodes(self):
        data = data_extraction.extract_linkedin_profile("any-url", mock=True)
        nodes = data_processing.split_profile_data(data)
        self.assertGreaterEqual(len(nodes), 1)

    def test_create_vector_database_and_verify(self):
        with patch.object(data_processing, "create_watsonx_embedding", fake_emb):
            data = data_extraction.extract_linkedin_profile("any-url", mock=True)
            nodes = data_processing.split_profile_data(data)
            index = data_processing.create_vector_database(nodes)
            self.assertIsNotNone(index)
            self.assertTrue(data_processing.verify_embeddings(index))


class TestQueryEngine(unittest.TestCase):
    def setUp(self):
        with patch.object(data_processing, "create_watsonx_embedding", fake_emb):
            data = data_extraction.extract_linkedin_profile("any-url", mock=True)
            nodes = data_processing.split_profile_data(data)
            self.index = data_processing.create_vector_database(nodes)

    def test_generate_initial_facts_returns_text(self):
        with patch.object(query_engine, "create_watsonx_llm", fake_llm), \
             patch.object(data_processing, "create_watsonx_embedding", fake_emb):
            facts = query_engine.generate_initial_facts(self.index)
            self.assertIsInstance(facts, str)
            self.assertNotEqual(facts, "Failed to generate initial facts.")

    def test_answer_user_query_returns_response(self):
        with patch.object(query_engine, "create_watsonx_llm", fake_llm), \
             patch.object(data_processing, "create_watsonx_embedding", fake_emb):
            resp = query_engine.answer_user_query(self.index, "Where does this person work?")
            self.assertTrue(hasattr(resp, "response"))


class TestGradioApp(unittest.TestCase):
    def test_interface_constructs(self):
        import gradio as gr
        import app
        demo = app.create_gradio_interface()
        self.assertIsInstance(demo, gr.Blocks)

    def test_chat_without_session_gives_helpful_message(self):
        import app
        history = app.chat_with_profile(None, "Who is this?", [])
        self.assertIn("No profile loaded", history[-1][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
