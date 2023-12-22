import json
import logging
from typing import List, Optional

import cattrs
from attr import define
from aiohttp import web
import socketio
from cattrs import structure, unstructure
from langchain.chat_models.base import BaseLanguageModel

import fable_saga

logger = logging.getLogger(__name__)

"""
Sets up a server that can be used to generate actions for SAGA. Either HTTP or socketio can be used.

For HTTP, make a POST request to the server with a JSON body of the following format:
For socketio, send a message with the following format to the "generate-actions" event:
{
  "context": "You are a mouse",
  "skills": [{
    "name": "goto",
    "description": "go somewhere",
    "parameters": {
      "location": "<str: where you want to go>"
    }
  }]
}

Response:




"""



@define(slots=True)
class ActionsRequest:
    """Request to generate actions."""
    context: str
    skills: List[fable_saga.Skill] = {}
    retries: int = 0
    verbose: bool = False
    reference: Optional[str] = None


@define(slots=True)
class ActionsResponse:
    """Response from generating actions."""
    actions: Optional[fable_saga.GeneratedActions] = None
    error: str = None
    reference: Optional[str] = None


class SagaServer:
    """ Server for SAGA. """
    def __init__(self, llm: BaseLanguageModel = None):
        super().__init__()
        if llm is None:
            llm = fable_saga.ChatOpenAI()
        self.agent = fable_saga.Agent(llm)

    async def generate_actions(self, req: ActionsRequest) -> ActionsResponse:
        # Generate actions
        try:
            assert isinstance(req, ActionsRequest), f"Invalid request type: {type(req)}"
            actions = await self.agent.generate_actions(req.context, req.skills, req.retries, req.verbose)
            response = ActionsResponse(actions=actions, reference=req.reference)
            if actions.error is not None:
                response.error = f"Generation Error: {actions.error}"
            return response
        except Exception as e:
            logger.error(str(e))
            return ActionsResponse(actions=None, error=str(e), reference=req.reference)


if __name__ == '__main__':

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, help="Type of server to run.", choices=["socketio", "http"])
    parser.add_argument('--host', type=str, default='localhost', help='Host to listen on')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--cors', type=str, default='*', help='CORS origin')
    args = parser.parse_args()

    # Create common server objects
    server = SagaServer()
    app = web.Application()

    # Create socketio server
    if args.type == 'socketio':
        sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins=args.cors)
        sio.attach(app)

        @sio.event
        def connect(sid, environ):
            logger.info("connect:" + sid)

        @sio.event
        def disconnect(sid):
            logger.info("disconnect:" + sid)

        @sio.on('*')
        def catch_all(event, sid, *data):
            """Catch all unhandled events that have one message. (common)"""
            logger.error(f"Unhandled event: {event} {sid} {data}")

        @sio.on('generate-actions')
        async def generate_actions(sid, message_data: str):
            logger.debug(f"Request from {sid}: {message_data}")
            try:
                data = json.loads(message_data)
                request = structure(data, ActionsRequest)
                actions = await server.generate_actions(request)
                response = unstructure(actions)
                logger.debug(f"Response: {response}")
                return response
            except cattrs.errors.ClassValidationError as e:
                error = f"Error validating request: {json.dumps(cattrs.transform_error(e))}"
            except Exception as e:
                error = str(e)
            logger.error(error)
            return unstructure(ActionsResponse(actions=None, error=error))

    # Create HTTP server
    elif args.type == 'http':
        """HTTP server
        Make a POST request to the server with a JSON body of the following format:
        {
          "type" : "generate-actions",
          "data" : {
            "context": "You are a mouse",
            "skills": [{
              "skill": "goto",
              "parameters": {
                "location": "<str: where you want to go>"
              }
            }]
          }
        }
        
        Response:
        {
          "options": [
            {
              "skill": "goto",
              "parameters": {
                "location": "cheese pantry"
              }
            },
            ...
          ],
          "scores": [
            0.8,
            ...
          ]
        }
        """


        async def generate_actions(request):
            """Handle POST requests to the server."""
            logger.debug(f"Request : {request}")
            params = await request.json()
            try:
                actions_req: ActionsRequest = structure(params, ActionsRequest)
                actions_response = await server.generate_actions(actions_req)
                response = unstructure(actions_response)
                logger.debug(f"Response: {response}")
                return web.json_response(response)
            except cattrs.errors.ClassValidationError as e:
                error = f"Error validating request: {json.dumps(cattrs.transform_error(e))}"
            except Exception as e:
                error = str(e)
            logger.error(error)
            return web.json_response(unstructure(ActionsResponse(actions=None, error=error, reference=params.get('reference'))))


        # Listen for POST requests on the root path
        app.add_routes([web.post('/', generate_actions)])

    else:
        raise ValueError("Invalid server type: " + args.type)

    # Setup logging
    formatter = logging.Formatter('%(asctime)s - saga.server - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Run server
    web.run_app(app, host=args.host, port=args.port)
