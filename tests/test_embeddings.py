import asyncio
import json
import uuid
from functools import partial
from typing import List

import numpy
import pytest
from langchain.embeddings.fake import DeterministicFakeEmbedding
from langchain.schema.embeddings import Embeddings

import fable_saga.embeddings as saga_embed


@pytest.fixture
def fake_embedding_model():
    class FakeAsyncEmbeddingModel(DeterministicFakeEmbedding):

        async def aembed_documents(self, texts: List[str]):
            func = partial(self.embed_documents, texts)
            return await asyncio.get_event_loop().run_in_executor(None, func)

        async def aembed_query(self, text: str):
            func = partial(self.embed_query, text)
            return await asyncio.get_event_loop().run_in_executor(None, func)

        def _select_relevance_score_fn(self, query: str):
            return lambda x: 0.5

    return FakeAsyncEmbeddingModel(size=1536)


@pytest.fixture
def fake_documents():
    data = json.load(open("examples/embedding_documents.json"))
    documents = [saga_embed.Document(**doc) for doc in data["documents"]]
    return documents


class TestEmbeddingsAgent:

    def test_init(self, fake_embedding_model: Embeddings):
        agent = saga_embed.EmbeddingAgent(fake_embedding_model)
        assert agent._embeddings_model == fake_embedding_model
        assert isinstance(agent._storage, saga_embed.SimpleVectorStore)

    @pytest.mark.asyncio
    async def test_embed_documents(
        self, fake_embedding_model: Embeddings, fake_documents
    ):
        agent = saga_embed.EmbeddingAgent(fake_embedding_model)
        embeddings = await agent.embed_documents([d.text for d in fake_documents])
        assert len(embeddings) == len(fake_documents)
        assert len(embeddings[0]) == 1536
        assert type(embeddings[0][0]) is numpy.float64

    @pytest.mark.asyncio
    async def test_embed_query(self, fake_embedding_model: Embeddings):
        agent = saga_embed.EmbeddingAgent(fake_embedding_model)
        embedding = await agent.embed_query("query")
        assert len(embedding) == 1536
        assert type(embedding[0]) is numpy.float64

    @pytest.mark.asyncio
    async def test_store_documents(
        self, fake_embedding_model: Embeddings, fake_documents
    ):
        agent = saga_embed.EmbeddingAgent(fake_embedding_model)
        ids = await agent.store_documents(fake_documents)
        # Check that the count of ids is the same as the count of documents.
        assert len(ids) == len(fake_documents)

        # Check that the ids are valid UUIDs.
        assert type(ids[0]) is str
        some_id = uuid.UUID(ids[0])
        assert type(some_id) is uuid.UUID

        # Check that the ids are unique.
        assert len(set(ids)) == len(ids)

    @pytest.mark.asyncio
    async def test_find_similar(self, fake_embedding_model: Embeddings, fake_documents):
        agent = saga_embed.EmbeddingAgent(fake_embedding_model)
        ids = await agent.store_documents(fake_documents)

        # Add the ids to the fake document's metadata since search seems to add the ids to the metadata itself.
        for idx, new_id in enumerate(ids):
            fake_documents[idx].metadata["id"] = new_id

        # Perform a real search (with fake embeddings).
        similar = await agent.find_similar(fake_documents[1].text, k=2)
        assert len(similar) == 2

        # Check the (fake) similarity scores. Embeddings are fake but deterministic, so we can check the scores here.
        doc, score = similar[0]
        assert doc.text == fake_documents[1].text
        assert doc.metadata == fake_documents[1].metadata
        assert score == pytest.approx(1.0, rel=1e-3)

        doc, score = similar[1]
        assert doc.text == fake_documents[0].text
        assert doc.metadata == fake_documents[0].metadata
        assert score == pytest.approx(0.5098, rel=1e-3)
