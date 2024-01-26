import base64
from typing import Dict
from unittest.mock import AsyncMock

import pytest
from cattr import unstructure

import fable_saga
from fable_saga import server as saga_server
from test_embeddings import fake_embedding_model, fake_documents
from test_saga import fake_actions_llm, fake_skills, fake_actions_request, fake_conversation_llm, fake_conversation_request


class TestSagaServer:

    def test_init(self, fake_actions_llm):
        server = saga_server.SagaServer(llm=fake_actions_llm)
        assert server.agent._llm == fake_actions_llm

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_actions_llm, fake_actions_request):
        server = saga_server.SagaServer(llm=fake_actions_llm)
        response = await server.generate_actions(fake_actions_request)

        # The response is a valid ActionsResponse
        # Note: we don't throw exceptions in the server, but return the error in the response.
        assert response.error is None

        # The reference is the same as the request.
        assert response.reference == fake_actions_request.reference

        # Sanity check that the skills are the same in our fake_llm (test data ignores the requested skills here).
        assert isinstance(response.actions, fable_saga.GeneratedActions)
        options = response.actions.options
        assert len(options) == 2
        assert options[0].skill == "skill_2"
        assert options[1].skill == "skill_1"

    @pytest.mark.asyncio
    async def test_generate_conversation(self, fake_conversation_llm, fake_conversation_request):
        server = saga_server.SagaServer(llm=fake_conversation_llm)
        response = await server.generate_conversation(fake_conversation_request)

        # The response is a valid ActionsResponse
        # Note: we don't throw exceptions in the server, but return the error in the response.
        assert response.error is None

        # The reference is the same as the request.
        assert response.reference == fake_conversation_request.reference

        # Sanity check that the skills are the same in our fake_llm (test data ignores the requested skills here).
        assert isinstance(response.conversation, fable_saga.GeneratedConversation)

        # Validate conversation data
        conversation = response.conversation.conversation
        assert len(conversation) == 2
        assert conversation[0].persona_guid == "person_a"
        assert conversation[0].dialogue == "person_a_dialogue"
        assert conversation[1].persona_guid == "person_b"
        assert conversation[1].dialogue == "person_b_dialogue"

    @pytest.mark.asyncio
    async def test_no_skills_error(self, fake_actions_llm, fake_actions_request):
        server = saga_server.SagaServer(llm=fake_actions_llm)

        # Pass an empty list of skills should raise an error.
        fake_actions_request.skills = []
        response = await server.generate_actions(fake_actions_request)
        assert response.error is not None
        assert response.actions is None
        assert response.reference == fake_actions_request.reference

        assert response.error == "Must provide at least one skill."

    @pytest.mark.asyncio
    async def test_malformed_skills_error(self, fake_actions_llm, fake_actions_request):
        server = saga_server.SagaServer(llm=fake_actions_llm)

        # Pass an empty list of skills should raise an error.
        fake_actions_request.skills = ["malformed"]
        response = await server.generate_actions(fake_actions_request)
        assert response.error is not None
        assert response.actions is None
        assert response.reference == fake_actions_request.reference

        assert response.error == "Must provide a list of Skill objects."

    @pytest.mark.asyncio
    async def test_malformed_response_error(self, fake_actions_llm, fake_actions_request):
        fake_actions_llm.responses = ["malformed"]
        server = saga_server.SagaServer(llm=fake_actions_llm)

        # A malformed response should raise an error.
        response = await server.generate_actions(fake_actions_request)
        assert response.actions is not None
        assert response.actions.error == ("No options found after 1 retries. Last error: Error decoding response:"
                                          " Expecting value: line 1 column 1 (char 0)")
        assert response.reference == fake_actions_request.reference
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

    @pytest.mark.asyncio
    async def test_generate_embeddings_error(self, fake_embedding_model):
        server = saga_server.EmbeddingsServer(embeddings=fake_embedding_model)

        # noinspection PyTypeChecker
        request = saga_server.EmbeddingsRequest(texts=1)
        response = await server.generate_embeddings(request)
        assert response.error == "'int' object is not iterable"

    @pytest.mark.asyncio
    async def test_add_documents(self, fake_embedding_model, fake_documents):
        server = saga_server.EmbeddingsServer(embeddings=fake_embedding_model)
        request = saga_server.AddDocumentsRequest(fake_documents)
        response = await server.add_documents(request)
        assert response.error is None
        assert len(response.guids) == 3

    @pytest.mark.asyncio
    async def test_find_similar(self, fake_embedding_model, fake_documents):
        server = saga_server.EmbeddingsServer(embeddings=fake_embedding_model)
        request = saga_server.AddDocumentsRequest(fake_documents)
        await server.add_documents(request)

        request = saga_server.FindSimilarRequest("test", k=2)
        response = await server.find_similar(request)
        assert response.error is None
        assert len(response.documents) == 2
        assert len(response.scores) == 2
        assert response.error is None


class TestGenericHandler:

    @pytest.mark.asyncio
    async def test_request_missing_key_error(self):
        missing_data = {}
        mock = AsyncMock()

        response: Dict = await saga_server.generic_handler(missing_data, saga_server.EmbeddingsRequest,
                                                           mock, saga_server.EmbeddingsResponse)
        assert response['error'] is not None
        assert response['error'] == 'Error validating request: ["required field missing @ $.texts"]'

    @pytest.mark.asyncio
    async def test_request_bad_json_sting_error(self):
        bad_json = 'asdf'
        mock = AsyncMock()

        response: Dict = await saga_server.generic_handler(bad_json, saga_server.EmbeddingsRequest,
                                                           mock, saga_server.EmbeddingsResponse)
        assert response['error'] is not None
        assert response['error'] == 'Error decoding JSON: Expecting value: line 1 column 1 (char 0)'

    @pytest.mark.asyncio
    async def test_request_error_extra_field(self):
        bad_data = {"texts": ["1", "2"], "extra": "extra"}

        mock = AsyncMock()

        response: Dict = await saga_server.generic_handler(bad_data, saga_server.EmbeddingsRequest,
                                                           mock, saga_server.EmbeddingsResponse)
        assert response['error'] is not None
        assert response['error'] == 'Error validating request: ["extra fields found (extra) @ $"]'

    @pytest.mark.asyncio
    async def test_processing_exception_error(self):
        async def fake_error(_):
            raise Exception("fake error")

        mock = AsyncMock(side_effect=fake_error)
        fake_data = {"texts": ["1", "2"]}

        response: Dict = await saga_server.generic_handler(fake_data, saga_server.EmbeddingsRequest,
                                                           mock, saga_server.EmbeddingsResponse)
        assert response['error'] == 'Error processing request: fake error'
        mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_skills):
        expected_request = saga_server.ActionsRequest(context="context", skills=fake_skills)
        fake_data = {"context": "context", "skills": unstructure(fake_skills)}

        async def fake_processor(_):
            return saga_server.ActionsResponse(
                actions=fable_saga.GeneratedActions(options=[fable_saga.Action("some_skill")], scores=[1]))

        mock = AsyncMock(side_effect=fake_processor)

        response: Dict = await saga_server.generic_handler(fake_data, saga_server.ActionsRequest,
                                                           mock, saga_server.ActionsResponse)
        # Check that the mock was called with the right data.
        mock.assert_called_once_with(expected_request)
        assert response['error'] is None
        assert response['actions']['options'] == [{'parameters': {}, 'skill': 'some_skill'}]
        assert response['actions']['scores'] == [1]
        assert response['reference'] is None

    @pytest.mark.asyncio
    async def test_generate_embeddings(self):
        fake_data = {"texts": ["1", "2"]}
        expected_request = saga_server.EmbeddingsRequest(**fake_data)

        async def fake_processor(_):
            return saga_server.EmbeddingsResponse(embeddings=["test"])

        mock = AsyncMock(side_effect=fake_processor)

        response: Dict = await saga_server.generic_handler(fake_data, saga_server.EmbeddingsRequest,
                                                           mock, saga_server.EmbeddingsResponse)
        # Check that the mock was called with the right data.
        mock.assert_called_once_with(expected_request)
        assert response['error'] is None
        assert response['embeddings'] == ["test"]
        assert response['reference'] is None

    @pytest.mark.asyncio
    async def test_add_documents(self, fake_documents):
        fake_data = {"documents": [unstructure(doc) for doc in fake_documents]}
        expected_request = saga_server.AddDocumentsRequest(documents=fake_documents)

        async def fake_processor(_):
            return saga_server.AddDocumentsResponse(guids=["1"])

        mock = AsyncMock(side_effect=fake_processor)

        response: Dict = await saga_server.generic_handler(fake_data, saga_server.AddDocumentsRequest,
                                                           mock, saga_server.AddDocumentsResponse)
        # Check that the mock was called with the right data.
        mock.assert_called_once_with(expected_request)
        assert response['error'] is None
        assert response['guids'] == ["1"]
        assert response['reference'] is None

    @pytest.mark.asyncio
    async def test_find_similar(self, fake_documents):
        fake_data = {"query": "test", "k": 2}
        expected_request = saga_server.FindSimilarRequest(query="test", k=2)

        async def fake_processor(_):
            # noinspection PyTestUnpassedFixture
            return saga_server.FindSimilarResponse(documents=fake_documents, scores=[1, 1, 1])

        mock = AsyncMock(side_effect=fake_processor)

        response: Dict = await saga_server.generic_handler(fake_data, saga_server.FindSimilarRequest,
                                                           mock, saga_server.FindSimilarResponse)
        # Check that the mock was called with the right data.
        mock.assert_called_once_with(expected_request)
        assert response['error'] is None
        assert response['documents'] == unstructure(fake_documents)
        assert response['scores'] == [1, 1, 1]
        assert response['reference'] is None
