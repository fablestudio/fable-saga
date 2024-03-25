import base64
import json
import logging
import struct
from typing import List, Optional, Type, Dict, Union

import cattrs
import socketio
from aiohttp import web, WSMsgType
from attr import define
from langchain.llms.base import BaseLanguageModel

import fable_saga
from fable_saga.conversations import GeneratedConversation, ConversationAgent
from fable_saga.embeddings import Document, EmbeddingAgent
from fable_saga.actions import Skill, ActionsAgent

logger = logging.getLogger(__name__)

# module level converter to convert between objects and dicts.
converter = cattrs.Converter(forbid_extra_keys=False)

"""
Sets up a server that can be used to generate actions for SAGA. Either HTTP or socketio can be used.
"""


@define(slots=True)
class ActionsRequest:
    """Request to generate actions."""

    context: str
    skills: List[fable_saga.actions.Skill]
    retries: int = 0
    verbose: bool = False
    reference: Optional[str] = None
    model: Optional[str] = None


@define(slots=True)
class ActionsResponse:
    """Response from generating actions."""

    actions: Optional[fable_saga.actions.GeneratedActions] = None
    error: Optional[str] = None
    reference: Optional[str] = None


@define(slots=True)
class ConversationRequest:
    """Request to generate a conversation."""

    context: str
    persona_guids: List[str]
    retries: int = 0
    verbose: bool = False
    reference: Optional[str] = None
    model: Optional[str] = None


@define(slots=True)
class ConversationResponse:
    """Response from generating a conversation."""

    conversation: Optional[GeneratedConversation] = None
    error: Optional[str] = None
    reference: Optional[str] = None


@define(slots=True)
class EmbeddingsRequest:
    """Request to generate embeddings."""

    texts: List[str]
    reference: Optional[str] = None


@define(slots=True)
class EmbeddingsResponse:
    """Response from generating embeddings."""

    embeddings: List[str] = (
        []
    )  # Actually list of List[float], but we pack 4 bytes per and then base64 encode them.
    error: Optional[str] = None
    reference: Optional[str] = None


@define(slots=True)
class AddDocumentsRequest:
    """Request to add documents."""

    documents: List[Document]
    reference: Optional[str] = None


@define(slots=True)
class AddDocumentsResponse:
    """Response from adding documents."""

    guids: List[str] = []
    error: Optional[str] = None
    reference: Optional[str] = None


@define(slots=True)
class FindSimilarRequest:
    """Request to find similar documents."""

    query: str
    k: int = 5
    reference: Optional[str] = None


@define(slots=True)
class FindSimilarResponse:
    """Response from finding similar documents."""

    documents: List[Document] = []
    scores: List[float] = []
    error: Optional[str] = None
    reference: Optional[str] = None


@define(slots=True)
class ErrorResponse:
    """Generic Error Response."""

    error: Optional[str] = None


class SagaServer:
    """Server for SAGA."""

    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__()
        self.agent = ActionsAgent(llm)

    async def generate_actions(self, req: ActionsRequest) -> ActionsResponse:
        # Generate actions
        try:
            assert isinstance(req, ActionsRequest), f"Invalid request type: {type(req)}"
            actions = await self.agent.generate_actions(
                req.context, req.skills, req.retries, req.verbose, req.model
            )
            response = ActionsResponse(actions=actions, reference=req.reference)
            if actions.error is not None:
                response.error = f"Generation Error: {actions.error}"
            return response
        except Exception as e:
            logger.error(str(e))
            return ActionsResponse(actions=None, error=str(e), reference=req.reference)


class ConversationServer:
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__()
        self.agent = ConversationAgent(llm)

    async def generate_conversation(
        self, req: ConversationRequest
    ) -> ConversationResponse:
        # Generate conversation
        try:
            assert isinstance(
                req, ConversationRequest
            ), f"Invalid request type: {type(req)}"
            conversation = await self.agent.generate_conversation(
                req.persona_guids, req.context, req.retries, req.verbose, req.model
            )
            response = ConversationResponse(
                conversation=conversation, reference=req.reference
            )
            if conversation.error is not None:
                response.error = f"Generation Error: {conversation.error}"
            return response
        except Exception as e:
            logger.error(str(e))
            return ConversationResponse(
                conversation=None, error=str(e), reference=req.reference
            )


class EmbeddingsServer:
    """Server for Embeddings."""

    def __init__(self, **kwargs):
        super().__init__()
        self.agent = EmbeddingAgent(**kwargs)

    async def generate_embeddings(self, req: EmbeddingsRequest) -> EmbeddingsResponse:
        # Generate embeddings
        try:
            assert isinstance(
                req, EmbeddingsRequest
            ), f"Invalid request type: {type(req)}"
            embeddings = await self.agent.embed_documents(req.texts)
            packed_embeddings = [
                base64.b64encode(struct.pack("!%sf" % len(e), *e)).decode("ascii")
                for e in embeddings
            ]
            response = EmbeddingsResponse(
                embeddings=packed_embeddings, reference=req.reference
            )
            return response
        except Exception as e:
            logger.error(str(e))
            return EmbeddingsResponse(
                embeddings=[], error=str(e), reference=req.reference
            )

    async def add_documents(self, req: AddDocumentsRequest) -> AddDocumentsResponse:
        # Add documents
        try:
            assert isinstance(
                req, AddDocumentsRequest
            ), f"Invalid request type: {type(req)}"
            guids = await self.agent.store_documents(req.documents)
            response = AddDocumentsResponse(guids=guids, reference=req.reference)
            return response
        except Exception as e:
            logger.error(str(e))
            return AddDocumentsResponse(guids=[], error=str(e), reference=req.reference)

    async def find_similar(self, req: FindSimilarRequest) -> FindSimilarResponse:
        # Find similar documents
        try:
            assert isinstance(
                req, FindSimilarRequest
            ), f"Invalid request type: {type(req)}"
            results = await self.agent.find_similar(req.query, req.k)
            documents, scores = zip(*results)
            response = FindSimilarResponse(
                documents=documents, scores=scores, reference=req.reference  # type: ignore
            )
            return response
        except Exception as e:
            logger.error(str(e))
            return FindSimilarResponse(
                documents=[], scores=[], error=str(e), reference=req.reference
            )


async def generic_handler(
    data: Union[str, Dict], request_type: Type, process_function, response_type: Type
):
    try:
        if isinstance(data, str):
            data = json.loads(data)
        # noinspection PyTypeChecker
        request = converter.structure(data, request_type)
        result = await process_function(request)
        assert isinstance(result, response_type), (
            f"Invalid response type: {type(result)},"
            f" expected instance of {response_type}"
        )
        response = converter.unstructure(result)
        logger.debug(f"Response: {response}")
        return response
    except json.decoder.JSONDecodeError as e:
        error = f"Error decoding JSON: {str(e)}"
    except cattrs.errors.ClassValidationError as e:
        error = f"Error validating request: {json.dumps(cattrs.transform_error(e))}"
    except Exception as e:
        error = f"Error processing request: {str(e)}"
    logger.error(error)
    response = response_type(error=error)
    output = converter.unstructure(response)
    return output


if __name__ == "__main__":

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="Type of server to run.",
        choices=["socketio", "http", "websockets"],
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host to listen on"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--cors", type=str, default=None, help="CORS origin")
    args = parser.parse_args()

    # Create common server objects
    # Note: This is where you could override the LLM by passing the llm parameter to SagaServer.
    saga_server = SagaServer()
    embeddings_server = EmbeddingsServer()
    conversation_server = ConversationServer()

    app = web.Application()

    # Create socketio server
    if args.type == "socketio":
        if args.cors is None:
            args.cors = "*"
        sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins=args.cors)
        sio.attach(app)

        @sio.event
        def connect(sid, _):
            logger.info("connect:" + sid)

        @sio.event
        def disconnect(sid):
            logger.info("disconnect:" + sid)

        @sio.on("*")
        def catch_all(event, sid, *data):
            """Catch all unhandled events that have one message. (common)"""
            logger.error(f"Unhandled event: {event} {sid} {data}")

        @sio.on("generate-actions")
        async def generate_actions(sid, message_str: str):
            logger.debug(f"Request from {sid}: {message_str}")
            return await generic_handler(
                message_str,
                ActionsRequest,
                saga_server.generate_actions,
                ActionsResponse,
            )

        @sio.on("generate-conversation")
        async def generate_conversation(sid, message_str: str):
            logger.debug(f"Request from {sid}: {message_str}")
            return await generic_handler(
                message_str,
                ConversationRequest,
                conversation_server.generate_conversation,
                ConversationResponse,
            )

        @sio.on("generate-embeddings")
        async def generate_embeddings(sid, message_str: str):
            logger.debug(f"Request from {sid}: {message_str}")
            return await generic_handler(
                message_str,
                EmbeddingsRequest,
                embeddings_server.generate_embeddings,
                EmbeddingsResponse,
            )

        @sio.on("add-documents")
        async def add_documents(sid, message_str: str):
            logger.debug(f"Request from {sid}: {message_str}")
            return await generic_handler(
                message_str,
                AddDocumentsRequest,
                embeddings_server.add_documents,
                AddDocumentsResponse,
            )

        @sio.on("find-similar")
        async def find_similar(sid, message_str: str):
            logger.debug(f"Request from {sid}: {message_str}")
            return await generic_handler(
                message_str,
                FindSimilarRequest,
                embeddings_server.find_similar,
                FindSimilarResponse,
            )

    # Create HTTP server
    elif args.type == "http":
        """HTTP server
        Make a POST request to the server with a JSON body message (See README.md for details).
        """

        routes = web.RouteTableDef()

        @routes.post("/generate-actions")
        async def generate_actions(request):
            """Handle POST requests to the server."""
            message_str = await request.text()
            logger.debug(f"Request: {message_str}")
            response = generic_handler(
                message_str,
                ActionsRequest,
                saga_server.generate_actions,
                ActionsResponse,
            )
            return web.json_response(response)

        @routes.post("/generate-conversation")
        async def generate_conversation(request):
            """Handle POST requests to the server."""
            message_str = await request.text()
            logger.debug(f"Request: {message_str}")
            response = generic_handler(
                message_str,
                ConversationRequest,
                conversation_server.generate_conversation,
                ConversationResponse,
            )
            return web.json_response(response)

        @routes.post("/generate-embeddings")
        async def generate_embeddings(request):
            """Handle POST requests to the server."""
            message_str = await request.text()
            logger.debug(f"Request: {message_str}")
            response = generic_handler(
                message_str,
                EmbeddingsRequest,
                embeddings_server.generate_embeddings,
                EmbeddingsResponse,
            )
            return web.json_response(response)

        @routes.post("/add-documents")
        async def add_documents(request):
            """Handle POST requests to the server."""
            message_str = await request.text()
            logger.debug(f"Request: {message_str}")
            response = generic_handler(
                message_str,
                AddDocumentsRequest,
                embeddings_server.add_documents,
                AddDocumentsResponse,
            )
            return web.json_response(response)

        @routes.post("/find-similar")
        async def find_similar(request):
            """Handle POST requests to the server."""
            message_str = await request.text()
            logger.debug(f"Request: {message_str}")
            response = generic_handler(
                message_str,
                FindSimilarRequest,
                embeddings_server.find_similar,
                FindSimilarResponse,
            )
            return web.json_response(response)

        # Listen for POST requests on the root path
        app.add_routes([web.post("/", generate_actions)])

    elif args.type == "websockets":

        from aiohttp import WSMessage

        async def websocket_handler(request):
            logger.info("Websocket connection starting")
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            logger.info("Websocket connection ready")

            async for msg in ws:  # type: WSMessage
                try:
                    if msg.type == WSMsgType.ERROR:
                        print("ws connection closed with exception %s" % ws.exception())
                    elif msg.type == WSMsgType.TEXT:
                        print(msg.data)
                        if msg.data == "close":
                            await ws.close()
                        else:
                            try:
                                data = json.loads(msg.data)
                                request_type = data.get("request-type")
                                request_data = data.get("request-data")

                                if request_type == "generate-actions":
                                    response = await generic_handler(
                                        request_data,
                                        ActionsRequest,
                                        saga_server.generate_actions,
                                        ActionsResponse,
                                    )
                                elif request_type == "generate-conversation":
                                    response = await generic_handler(
                                        request_data,
                                        ConversationRequest,
                                        conversation_server.generate_conversation,
                                        ConversationResponse,
                                    )
                                elif request_type == "generate-embeddings":
                                    response = await generic_handler(
                                        request_data,
                                        EmbeddingsRequest,
                                        embeddings_server.generate_embeddings,
                                        EmbeddingsResponse,
                                    )
                                elif request_type == "add-documents":
                                    response = await generic_handler(
                                        request_data,
                                        AddDocumentsRequest,
                                        embeddings_server.add_documents,
                                        AddDocumentsResponse,
                                    )
                                elif request_type == "find-similar":
                                    response = await generic_handler(
                                        request_data,
                                        FindSimilarRequest,
                                        embeddings_server.find_similar,
                                        FindSimilarResponse,
                                    )
                                else:
                                    error = f"Invalid request-type: {request_type}"
                                    logger.error(error)
                                    response = converter.unstructure(
                                        ErrorResponse(error=error)
                                    )
                            except Exception as e:
                                logger.error(str(e))
                                response = converter.unstructure(
                                    ErrorResponse(error=str(e))
                                )
                            await ws.send_json(response)
                except Exception as e:
                    logger.error(str(e))
                    continue
            print("Websocket connection closed")
            return ws

        # Use the root path for websocket connection.
        app.router.add_route("GET", "/", websocket_handler)

    else:
        raise ValueError("Invalid server type: " + args.type)

    # Setup logging
    formatter = logging.Formatter(
        "%(asctime)s - saga.server - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Run server
    web.run_app(app, host=args.host, port=args.port)
