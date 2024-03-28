import base64
from json import JSONDecodeError
from typing import Dict, cast
from unittest.mock import AsyncMock, patch

import cattrs
import pytest
from cattr import unstructure

import fable_saga
import fable_saga.actions
import fable_saga.conversations
import fable_saga.embeddings
import fable_saga.server
from fable_saga import server

# This line is needed to import BEFORE the other fixtures because otherwise they aren't found for some reason.
from . import (
    fake_actions_llm,
    fake_conversation_llm,
    fake_embedding_model,
)
from . import (
    fake_actions_agent,
    fake_conversation_agent,
    fake_embedding_agent,
    fake_skills,
    fake_actions_request,
    fake_conversation_request,
    fake_documents,
    EndpointMocker,
)


class TestActionsEndpoint:

    def test_init(self, fake_actions_agent):
        endpoint = server.ActionsEndpoint(fake_actions_agent)
        assert endpoint.agent == fake_actions_agent

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_actions_agent, fake_actions_request):
        endpoint = server.ActionsEndpoint(fake_actions_agent)
        response = await endpoint.handle_request(fake_actions_request)

        # The response is a valid ActionsResponse
        # Note: we don't throw exceptions in the endpoint, but return the error in the response.
        assert response.error is None

        # The reference is the same as the request.
        assert response.reference == fake_actions_request.reference

        # Sanity check that the skills are the same in our fake_llm (test data ignores the requested skills here).
        assert isinstance(response.actions, fable_saga.actions.GeneratedActions)
        options = response.actions.options
        assert len(options) == 2
        assert options[0].skill == "skill_2"
        assert options[1].skill == "skill_1"

    @pytest.mark.asyncio
    async def test_no_skills_error(self, fake_actions_agent, fake_actions_request):
        endpoint = server.ActionsEndpoint(fake_actions_agent)

        # Pass an empty list of skills should raise an error.
        fake_actions_request.skills = []
        with pytest.raises(AssertionError, match="Must provide at least one skill."):
            await endpoint.handle_request(fake_actions_request)

    @pytest.mark.asyncio
    async def test_malformed_skills_error(
        self, fake_actions_agent, fake_actions_request
    ):
        endpoint = server.ActionsEndpoint(fake_actions_agent)

        # Pass an empty list of skills should raise an error.
        fake_actions_request.skills = ["malformed"]
        with pytest.raises(
            AssertionError, match="Must provide a list of Skill objects."
        ):
            await endpoint.handle_request(fake_actions_request)


class TestConversationEndpoint:

    @pytest.mark.asyncio
    async def test_generate_conversation(
        self, fake_conversation_agent, fake_conversation_request
    ):
        endpoint = server.ConversationEndpoint(fake_conversation_agent)
        response = await endpoint.handle_request(fake_conversation_request)

        # The response is a valid ActionsResponse
        # Note: we don't throw exceptions in the endpoint, but return the error in the response.
        assert response.error is None

        # The reference is the same as the request.
        assert response.reference == fake_conversation_request.reference

        # Sanity check that the skills are the same in our fake_llm (test data ignores the requested skills here).
        assert isinstance(
            response.conversation, fable_saga.conversations.GeneratedConversation
        )

        # Validate conversation data
        conversation = response.conversation.conversation
        assert len(conversation) == 2
        assert conversation[0].persona_guid == "person_a"
        assert conversation[0].dialogue == "person_a_dialogue"
        assert conversation[1].persona_guid == "person_b"
        assert conversation[1].dialogue == "person_b_dialogue"


class TestEmbeddingServer:
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, fake_embedding_agent):
        endpoint = server.GenerateEmbeddingsEndpoint(fake_embedding_agent)
        request = server.EmbeddingsRequest(texts=["test1", "test2"])
        response = await endpoint.handle_request(request)
        assert response.error is None

        # Check the number of embeddings is a list of 2.
        assert response.embeddings is not None
        assert len(response.embeddings) == 2

        # The embedding is a valid base64 string, and it's the right length of float32s (4 bytes).
        base64_embedding = base64.b64decode(response.embeddings[0])
        assert len(base64_embedding) == 1536 * 4

    @pytest.mark.asyncio
    async def test_generate_embeddings_error(self, fake_embedding_agent):
        endpoint = server.GenerateEmbeddingsEndpoint(fake_embedding_agent)

        # noinspection PyTypeChecker
        request = server.EmbeddingsRequest(texts=1)  # type: ignore
        with pytest.raises(TypeError, match="'int' object is not iterable"):
            response = await endpoint.handle_request(request)

    @pytest.mark.asyncio
    async def test_add_documents(self, fake_embedding_agent, fake_documents):
        endpoint = server.AddDocumentsEndpoint(fake_embedding_agent)
        request = server.AddDocumentsRequest(fake_documents)
        response = await endpoint.handle_request(request)
        assert response.error is None
        assert len(response.guids) == 3

    @pytest.mark.asyncio
    async def test_find_similar(self, fake_embedding_agent, fake_documents):
        add_docs_endpoint = server.AddDocumentsEndpoint(fake_embedding_agent)
        add_docs_request = server.AddDocumentsRequest(fake_documents)
        await add_docs_endpoint.handle_request(add_docs_request)

        find_similar_request = server.FindSimilarRequest("test", k=2)
        find_similar_endpoint = server.FindSimilarEndpoint(fake_embedding_agent)
        response = await find_similar_endpoint.handle_request(find_similar_request)
        assert response.error is None
        assert len(response.documents) == 2
        assert len(response.scores) == 2
        assert response.error is None


class TestGenericHandler:

    throw_exception_states = [True, False]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("throw_exceptions", throw_exception_states)
    async def test_throw_exceptions(
        self, throw_exceptions, fake_skills, fake_actions_agent
    ):
        # Test that the server can throw exceptions or not based on the throw_exceptions flag.
        with patch("fable_saga.server.throw_exceptions", throw_exceptions):
            fake_data = {"context": "context", "skills": unstructure(fake_skills)}
            assert server.throw_exceptions == throw_exceptions

            async def raise_exception(*_) -> server.ActionsResponse:
                raise Exception("fake error")

            with EndpointMocker(
                server.ActionsEndpoint, override_handler=raise_exception
            ) as mock:
                endpoint = mock(fake_actions_agent)
                if throw_exceptions:
                    assert server.throw_exceptions is True
                    with pytest.raises(Exception, match="fake error"):
                        await server.generic_handler(
                            fake_data,
                            endpoint,
                        )
                else:
                    assert server.throw_exceptions is False
                    response: Dict = await server.generic_handler(
                        fake_data,
                        endpoint,
                    )
                    assert response["error"] == "Error processing request: fake error"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("throw_exceptions", throw_exception_states)
    async def test_request_missing_key_error(
        self, throw_exceptions: bool, fake_actions_agent
    ):
        with patch("fable_saga.server.throw_exceptions", throw_exceptions):
            missing_data: Dict = {}
            with EndpointMocker(server.ActionsEndpoint) as mock:
                endpoint = mock(fake_actions_agent)
                expected_error = 'Error validating request: ["required field missing @ $.context", "required field missing @ $.skills"]'

                if server.throw_exceptions:
                    with pytest.raises(
                        cattrs.errors.ClassValidationError,
                    ):
                        await server.generic_handler(
                            missing_data,
                            endpoint,
                        )
                else:
                    response: Dict = await server.generic_handler(
                        missing_data,
                        endpoint,
                    )
                    assert response["error"] is not None
                    assert response["error"] == expected_error

    @pytest.mark.asyncio
    @pytest.mark.parametrize("throw_exceptions", throw_exception_states)
    async def test_request_bad_json_sting_error(
        self, throw_exceptions, fake_actions_agent
    ):
        with patch("fable_saga.server.throw_exceptions", throw_exceptions):
            server.throw_exceptions = throw_exceptions
            bad_json = "asdf"
            with EndpointMocker(server.ActionsEndpoint) as mock:
                endpoint = mock(fake_actions_agent)

                if server.throw_exceptions:
                    with pytest.raises(JSONDecodeError):
                        await server.generic_handler(
                            bad_json,
                            endpoint,
                        )
                else:
                    response: Dict = await server.generic_handler(
                        bad_json,
                        endpoint,
                    )
                    assert response["error"] is not None
                    assert (
                        response["error"]
                        == "Error decoding JSON: Expecting value: line 1 column 1 (char 0)"
                    )

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_skills, fake_actions_agent):
        server.throw_exceptions = True
        expected_request = server.ActionsRequest(context="context", skills=fake_skills)
        fake_data = {"context": "context", "skills": unstructure(fake_skills)}

        async def fake_handler(*args) -> server.ActionsResponse:
            return server.ActionsResponse(
                actions=fable_saga.actions.GeneratedActions(
                    options=[fable_saga.actions.Action("some_skill")], scores=[1]
                )
            )

        with EndpointMocker(
            server.ActionsEndpoint, override_handler=fake_handler
        ) as fake_endpoint_class:
            fake_endpoint = fake_endpoint_class(fake_actions_agent)

            response: Dict = await server.generic_handler(
                fake_data,
                fake_endpoint,
            )

            spy = cast(AsyncMock, fake_endpoint.handle_request)
            # Check that the mock was called with the right data.
            spy.assert_called_once_with(fake_endpoint, expected_request)
        assert response["error"] is None
        assert response["actions"]["options"] == [
            {"parameters": {}, "skill": "some_skill"}
        ]
        assert response["actions"]["scores"] == [1]
        assert response["reference"] is None

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, fake_embedding_agent):
        fake_data = {"texts": ["1", "2"]}
        expected_request = server.EmbeddingsRequest(texts=fake_data["texts"])

        async def fake_handler(*_) -> server.EmbeddingsResponse:
            return server.EmbeddingsResponse(embeddings=["test"])

        with EndpointMocker(
            server.GenerateEmbeddingsEndpoint, override_handler=fake_handler
        ) as fake_endpoint_class:
            fake_endpoint = fake_endpoint_class(fake_embedding_agent)

            response: Dict = await server.generic_handler(
                fake_data,
                fake_endpoint,
            )
            # Check that the mock was called with the right data.
            spy = cast(AsyncMock, fake_endpoint.handle_request)
            # Check that the mock was called with the right data.
            spy.assert_called_once_with(fake_endpoint, expected_request)

        assert response["error"] is None
        assert response["embeddings"] == ["test"]
        assert response["reference"] is None

    @pytest.mark.asyncio
    async def test_add_documents(self, fake_documents, fake_embedding_agent):
        fake_data = {"documents": [unstructure(doc) for doc in fake_documents]}
        expected_request = server.AddDocumentsRequest(documents=fake_documents)

        async def fake_handler(*_) -> server.AddDocumentsResponse:
            return server.AddDocumentsResponse(guids=["1"])

        with EndpointMocker(
            server.AddDocumentsEndpoint, override_handler=fake_handler
        ) as fake_endpoint_class:
            fake_endpoint = fake_endpoint_class(fake_embedding_agent)

            response: Dict = await server.generic_handler(
                fake_data,
                fake_endpoint,
            )
            spy = cast(AsyncMock, fake_endpoint.handle_request)
            # Check that the mock was called with the right data.
            spy.assert_called_once_with(fake_endpoint, expected_request)

        assert response["error"] is None
        assert response["guids"] == ["1"]
        assert response["reference"] is None

    @pytest.mark.asyncio
    async def test_find_similar(self, fake_documents, fake_embedding_agent):
        fake_data = {"query": "test", "k": 2}
        expected_request = server.FindSimilarRequest(query="test", k=2)

        async def fake_handler(*_) -> server.FindSimilarResponse:
            return server.FindSimilarResponse(documents=fake_documents, scores=[1, 1])

        with EndpointMocker(
            server.FindSimilarEndpoint, override_handler=fake_handler
        ) as fake_endpoint_class:
            fake_endpoint = fake_endpoint_class(fake_embedding_agent)

            response: Dict = await server.generic_handler(
                fake_data,
                fake_endpoint,
            )
            spy = cast(AsyncMock, fake_endpoint.handle_request)
            # Check that the mock was called with the right data.
            spy.assert_called_once_with(fake_endpoint, expected_request)
        assert response["error"] is None
        assert response["documents"] == unstructure(fake_documents)
        assert response["scores"] == [1, 1]
        assert response["reference"] is None
