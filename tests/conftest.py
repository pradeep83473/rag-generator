import hashlib
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import normalize
from app.main import create_app


class FakeProvider:
    """Deterministic offline stand-in for OpenAI.

    Embeddings are hashed bag-of-words vectors, so texts sharing words are
    similar. chat() records calls and cites source [1] (or answers from the
    condense prompt), which is enough to exercise the full pipeline.
    """

    embedding_model = "fake-embedding"
    DIM = 512

    def __init__(self):
        self.chat_calls: list[list[dict]] = []
        self.reply = "Answer based on the first source [1]."

    def embed(self, texts):
        matrix = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                matrix[row, h % self.DIM] += 1.0
        return normalize(matrix)

    def chat(self, messages):
        self.chat_calls.append(messages)
        if "standalone search query" in messages[0]["content"]:
            return "standalone: " + messages[-1]["content"].split("Latest question:")[-1].strip()
        return self.reply


@pytest.fixture
def settings(tmp_path):
    return Settings(openai_api_key="", data_dir=tmp_path, chunk_size=300, chunk_overlap=50)


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def client(settings, provider):
    return TestClient(create_app(settings, provider))
