import base64

import pytest

from fable_saga import server as saga_server
import fable_saga

from test_saga import fake_llm, fake_skills, fake_request
from test_embeddings import fake_embedding_model, fake_documents

class TestSagaServer:

    def test_init(self, fake_llm):
        server = saga_server.SagaServer(llm=fake_llm)
        assert server.agent._llm == fake_llm

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_llm, fake_request):
        server = saga_server.SagaServer(llm=fake_llm)
        response = await server.generate_actions(fake_request)

        # The response is a valid ActionsResponse
        # Note: we don't throw exceptions in the server, but return the error in the response.
        assert response.error is None

        # The reference is the same as the request.
        assert response.reference == fake_request.reference

        # Sanity check that the skills are the same in our fake_llm (test data ignores the requested skills here).
        assert isinstance(response.actions, fable_saga.GeneratedActions)
        options = response.actions.options
        assert len(options) == 2
        assert options[0].skill == "skill_2"
        assert options[1].skill == "skill_1"

    @pytest.mark.asyncio
    async def test_no_skills_error(self, fake_llm, fake_request):
        server = saga_server.SagaServer(llm=fake_llm)

        # Pass an empty list of skills should raise an error.
        fake_request.skills = []
        response = await server.generate_actions(fake_request)
        assert response.error is not None
        assert response.actions is None
        assert response.reference == fake_request.reference

        assert response.error == "Must provide at least one skill."


    @pytest.mark.asyncio
    async def test_malformed_skills_error(self, fake_llm, fake_request):
        server = saga_server.SagaServer(llm=fake_llm)

        # Pass an empty list of skills should raise an error.
        fake_request.skills = ["malformed"]
        response = await server.generate_actions(fake_request)
        assert response.error is not None
        assert response.actions is None
        assert response.reference == fake_request.reference

        assert response.error == "Must provide a list of Skill objects."

    @pytest.mark.asyncio
    async def test_malformed_response_error(self, fake_llm, fake_request):
        fake_llm.responses = ["malformed"]
        server = saga_server.SagaServer(llm=fake_llm)

        # A malformed response should raise an error.
        response = await server.generate_actions(fake_request)
        assert response.actions is not None
        assert response.actions.error == "No options found after 1 retries. Last error: Error decoding response: Expecting value: line 1 column 1 (char 0)"
        assert response.reference == fake_request.reference
        assert response.error is not None


class TestEmbeddingServer:
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, fake_embedding_model):
        server = saga_server.EmbeddingsServer(embeddings=fake_embedding_model)
        request = saga_server.EmbeddingsRequest(texts=["test1", "test2"])
        response = await server.generate_embeddings(request)
        assert response.error is None

        # Check the number of embeddings is a list of 2.
        assert response.embeddings is not None
        assert len(response.embeddings) == 2

        # The embedding is a valid base64 string, and it's the right length of float32s (4 bytes).
        base64_embedding = base64.b64decode(response.embeddings[0])
        assert len(base64_embedding) == 1536 * 4