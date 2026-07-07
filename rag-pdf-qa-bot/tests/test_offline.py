"""Offline test suite for the RAG QA bot — no watsonx credentials required.

The REAL pipeline components run: PyPDFLoader on a real PDF, the real
text splitter, a real Chroma vector store, a real retriever, and a real
RetrievalQA chain. Only the two watsonx factories are patched:
  - get_llm            -> FakeListLLM
  - watsonx_embedding  -> DeterministicFakeEmbedding

Run from the project root:  python tests/test_offline.py
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.llms.fake import FakeListLLM
from langchain_community.embeddings import DeterministicFakeEmbedding

import qabot

PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_document.pdf")

fake_llm = lambda: FakeListLLM(responses=["The revisit time is five days at the equator."] * 10)
fake_emb = lambda: DeterministicFakeEmbedding(size=64)


class TestPipelineComponents(unittest.TestCase):
    def test_document_loader_reads_pdf(self):
        docs = qabot.document_loader(PDF)
        self.assertGreaterEqual(len(docs), 1)
        self.assertIn("Sentinel-2", docs[0].page_content)

    def test_text_splitter_produces_chunks(self):
        docs = qabot.document_loader(PDF)
        chunks = qabot.text_splitter(docs)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all(len(c.page_content) <= 1000 for c in chunks))

    def test_vector_database_and_retriever(self):
        with patch.object(qabot, "watsonx_embedding", fake_emb):
            docs = qabot.document_loader(PDF)
            chunks = qabot.text_splitter(docs)
            vectordb = qabot.vector_database(chunks)
            results = vectordb.as_retriever().invoke("revisit time at the equator")
            self.assertGreaterEqual(len(results), 1)


class TestEndToEnd(unittest.TestCase):
    def test_retriever_qa_full_chain(self):
        with patch.object(qabot, "watsonx_embedding", fake_emb), \
             patch.object(qabot, "get_llm", fake_llm):
            answer = qabot.retriever_qa(PDF, "What is the revisit time of Sentinel-2?")
            self.assertIn("five days", answer)


class TestInterface(unittest.TestCase):
    def test_gradio_interface_constructs(self):
        import gradio as gr
        self.assertIsInstance(qabot.rag_application, gr.Interface)


if __name__ == "__main__":
    unittest.main(verbosity=2)
