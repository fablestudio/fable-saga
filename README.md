SAGA: Skill to Action Generation for Agents
===========================================

Join the Community
--------
> 🚨Private Beta Applications now open for the 3D environment that works with SAGA called Thistle Gulch - [Check out the announcement and apply](https://blog.fabledev.com/blog/beta-application-for-thistle-gulch-now-open)! In the meantime, try out the space-colony text based demo below.
> We're creating a dev-community around SAGA, Thistle Gulch, and Multi-Agent Simulations in general. [Reach out on twitter](https://twitter.com/frankcarey) if you're interested!

Demo Quickstart
-------------
1. Make sure you have git, python-3.10/11, and poetry installed and OPENAI key setup (see below).
2. Clone this repo and change the directory to it.
3. run `poetry install --all-extras --with test` to install dependencies.
4. start a poetry shell with `poetry shell` to make sure you are using the correct python version and have the correct environment variables set.
5. run `python -m fable_saga.demos.space_colony`

B.Y.O.S. (Bring Your Own Simulation)
--------
SAGA is a library that can be used to generate actions for any simulation. [Our blog post shows
SAGA being used in our "Thistle Gulch" simulation](https://blog.fabledev.com/), which is full on 3D interactive simulation, but to
get started, you can use the included Space Colony demo as a starting point.

The Space Colony demo is a simple text-based simulation of a spaceship with a crew of 5 agents. It's included in this 
repo to make it easier to test it out and have an example to work from. See the section below for more details on
running it, as well as how it uses SAGA to generate actions.

Using as a Server (HTTP, Websockets or SocketIO)
==================
SAGA can be used as a server that can be connected to via HTTP, Websockets or SocketIO. This is useful if you want to interface
with SAGA from a simulation that is not written in Python (like the Thistle Gulch demo). To start the server, run:

`python -m fable_saga.server --type <http, websockets or socketio> --port <defaults to 8080> --host <defaults to localhost> --cors <defaults to *>`

Note that the server is for demo purposes and probably not secure, so you should only run it locally or on a secure network.
If you wanted to write your own server, see the `fable_saga.server` module for details on how that's being done.

HTTP
----

`python -m fable_saga.server --type http`

The HTTP server is a REST API that accepts POST requests to the `/generate-actions` endpoint. The body of the request should be a JSON
object with the skills and context defined. You can optionally pass a "reference" that will be returned as well.

```
curl --request POST \
  --url http://localhost:8080/generate-actions \
  --header 'content-type: application/json' \
  --data '{
  "reference": "1234",
  "context": "You are a mouse",
  "skills": [{
    "name": "goto",
    "description": "go somewhere",
  	"parameters": {
      "location": "<str: where you want to go>"
    }
  }]
}'
```


The response looks like this:
```
{
  "actions": {
    "options": [
      {
        "skill": "goto",
        "parameters": {
          "location": "garden"
        }
      },
      ... all the other options
    ],
    "scores": [
      0.9,
      .. all the other scores
    ],
    "raw_prompt": <str: the raw prompt that was sent to the model, useful to debug.>,
    "raw_response": <str: the raw response from the model, useful if parsing failed and to debug.>,
    "llm_info": { ..this is specific to the model you are using, but openai models have this structure..
      "token_usage": { 
        "prompt_tokens": 164,
        "completion_tokens": 123,
        "total_tokens": 287
      },
      "model_name": "gpt-3.5-turbo"
    },
    "retries": 0, .. number of times the model was retried due to errors
    "error": null .. the error of the last try. null if no errors.
  },
  "error": null, .. the error either of the last try, or an error with the request itself.
  "reference": "1234" .. the reference you passed in
}
```

SocketIO
--------

`python -m fable_saga.server --type socketio`

The SocketIO server works similarly to the HTTP server, but it uses SocketIO instead.
You need to pass messages to the `generate-actions` event using the same JSON structure above, and the
response will be sent back to the same client that made the request in an async fashion.

The SocketIO server supports v4 of the SocketIO protocol, so you can use the SocketIO client of your choice.

When a client successfully connects to the server, you will see the following message:

`2023-12-22 12:50:07,972 - saga.server - INFO - connect:9CzowyUdrI1EOG0TAAAB`


Websockets
-------
`python -m fable_saga.server --type websockets`

Very similar to socketio (which uses websockets as part of its protocol). Websockets alone misses some features of socketio but works in a very similar way.
Simply connect to ws://127.0.0.1:8080 (by default) and wrap the json data in a request-type/request-data format, and you will get the same response.

```Json
{ "request-type": "generate-actions",
  "request-data": {
    "reference": 1234,
    "context": "You are a mouse",
    "skills": [{
      "name": "goto",
      "description": "go somewhere",
    	"parameters": { 
          "location": "<str: where you want to go>"
      	}}]}}
```

Embeddings Server (in progress)
-----------------

There's now an embeddings server that can be used to generate embeddings for any text. This is useful if you want to use
memories and the like in your context data, but don't want to send the entire text to the model. The embeddings server
provides embeddings, but will also store documents and find similar documents based on the embedding of a query you
provide. This is useful for finding similar memories, personas, etc.

Right now, there is only a single index that uses a simple in-memory numpy VectorStore to find similar documents. This is not
very scalable, but it's a good starting point and uses OpenAI's embeddings, which are very good. Both can be swapped
out for other systems supported by LangChain (see the `fable_saga.embeddings` module for details).

Also, right now there is only a single index, but it's possible to have multiple indexes soon, each with their own
embeddings and similar documents. This is useful if you want to have different indexes for different types of
documents, like personas, memories, etc or want to have different indexes for each character's memories.

It runs off the same default server so the following endpoints are available (http, websockets and socketio) and follow
the same pattern for communicating with each server type.

### Generate Embeddings
* http : `/generate-embeddings` - POST
* socketio : `generate-embeddings` - event
* websockets : `generate-embeddings` - request-type

Returns embeddings for the text you pass in as a list of strings.
The embeddings are returned as a list of lists of floats, but packed into a base64 string for transport.
You can unpack them by first decoding the base64 string to bytes and then for each 4 bytes,
unpack each them as a float (Big-Endian) and you will get the embeddings.
We do this because the embeddings are very large (~1536 floats in the case of OpenAI's ada-2),
so we want to avoid sending them directly as JSON.

```Json
 {
    "reference": 1234,
    "texts": ["Once upon a time"]
}
```

```Json
{
  "embeddings": [
    "PK5P2bykyIS7lO/OvD4okLw3bR87AXcVOxYt+/RvIXQsTuB4SQ8EiPqvO91cjx6pQK8MMwyuq..."
  ],
  "error": null,
  "reference": "1234"
}
```

### Add Documents
* http : `/add-documents` - POST
* socketio : `add-documents` - event
* websockets : `add-documents` - request-type

Adds documents to the index. The documents are stored in the VectorStore and the index is updated with the new documents.
The `text` field is required as that is what the embedding is based on, but you can add any other fields you want
to the document's `metadata` field. Note that `metadata.id` field will be added to the metadata automatically on retrieval.
The response includes the guids (ids) that are generated automatically by the VectorStore.

```Json
{
  "documents": [ {
    "text": "once upon a time",
    "metadata": {"some data": "I want to get back later"}
  }]
}
```

```Json
{
  "guids": [
    "d08d6696-7f48-4d62-8161-0e8c779e264b"
  ],
  "error": null,
  "reference": null
}
```

### Find Similar
* http : `/find-similar` - POST
* socketio : `find-similar` - event
* websockets : `find-similar` - request-type

Finds similar documents to the query you pass in. The query is a text string and the response is a list of documents. 
`k` is the number of similar documents to return. `k` must not be greater than the number of documents in the index.
The `scores` by default are using the sklearn(langchain integration) default which as of this writing is:
`score = 1 / exp(cosine_similarity - 1)`. This means that the scores ARE between 0 and 1, with 1 being the most similar,
but a cosine_similarity of 0 results in a score of 0.37 and a cosine_similarity of -1 results in a score of 0.135.
Expect to see many scores in the 0.8+ range when using the openAI embeddings no matter how different the documents are.

```Json
{
  "query": "in the beginning",
  "k": 1
}
```

```Json
{
  "documents": [
    {
      "text": "once upon a time",
      "metadata": {
        "id": "b1e0e032-439d-4255-8bb5-a02e0f2745f5"
      }
    }
  ],
  "scores": [
    0.8507239767422795
  ],
  "error": null,
  "reference": null
}
```


Space Colony Demo
==================

The Space Colony demo is a simple text-based simulation of a spaceship with a crew of 5 agents.
There are various locations on the ship that the agents can move to and objects they can interact
with. All of this data we call "Meta-Memories" and they formatted and provided to the `fable_saga.ActionsAgent`
when requesting actions.

Generating Actions from Skills
-------------
Right now the agents and their spaceship is very simple, but with the few actions they have, they
can generate a lot of interesting behavior. The actions they can take are defined by "Skills" which
are listed in the `fable_saga/demos/space_colony/resources/skills.yaml` file. Here are two of them:

```Yaml
- name: go_to
  description: "Go to a location in the world"
  parameters:
    destination: "<str: persona_guid, item_guid, or location.name to go to>"
    goal: "<str: goal of the movement>"

- name: converse_with
  description: "Walk to another character and talk to them"
  parameters:
    persona_guid: "<str: guid of the persona to converse with. You cannot talk to yourself.>"
    topic: "<str: topic of the conversation>"
    context: "<str: lots of helpful details the conversation generator can use to
     generate a conversation. It only has access to the context and the topic
      you provide, so be very detailed.>"
    goal: "<str: goal of the conversation>"
```
It's important to note that SAGA itself doesn't know how to drive the simulation or characters. That
is left to the simulation itself. SAGA only knows how to generate actions based on the skills you
provide it. You can add more skills to the demo for instance, and SAGA will be able to generate
actions for them, but you will have to implement the logic for those actions in the demo simulation.
You can also use the `fable_saga.ActionsAgent` class outside the demo to generate actions for your own sim. 
This demo just makes it easy to see how the agents use these skills to generate actions.

Skills are used by SAGA to generate Action options. In the demo, they are returned to you interactively
on the command line, where you can see the options and choose one for the agent to take. The highest
scored action is always the first one in the list, so typing "0" and ENTER repeatedly will always choose the
highest scored action.

```
========== nyah_kobari ===========
  ROLE: As the head of security, Nyah ensures the safety of the cargo and crew from pirates,
     smugglers, and other potential threats. She also serves as a tactical advisor
     to Captain Sundeep and leads boarding and inspection procedures.
  LOCATION: Mess Hall
  ACTION: Idle
  MEMORIES:
  * 0m ago: Moved from crew_quarters_corridor to mess_hall with goal socialize with the crew
..
Generating actions for nyah_kobari using model gpt-3.5-turbo-1106...
...
#0 -- converse_with (0.8)
  persona_guid: elara_sundeep
  topic: ship's route planning
  context: Discussing potential alternate routes and navigation strategies
    to avoid asteroid fields and optimize fuel usage
  goal: gather input for route planning
#1 -- interact (0.7)
  item_guid: crew_photo_wall
  interaction: view_photos
  goal: reflecting on past memories and relationships with the crew
> Choose an item for nyah_kobari...
```

Memories
-------------
As the agents move around the ship and interact with objects and other agents, they generate memories
of those interactions. These memories are formatted as a list when passed in as context data to
the `fable_saga.ActionsAgent` and `fable_saga.ConversationAgent` classese. The demo simulation keeps track of these memories, not SAGA.

The memories will continue to be generated as the agents interact with the world. This can add a lot of
tokens to the context data, so if you have any intention of running the demo for a long time, you will
want to implement some kind of memory management system to keep the context data from getting too large.

Since there isn't a memory management system in place, the demo simulation will eventually run out of context tokens
depending on the model you are using. The default model is `gpt-3.5-turbo-1106` which has a token limit of 4096 tokens.

Large Language Models
-------------
The demo uses the `gpt-3.5-turbo-1106` model from OpenAI by default. This model doesn't produce the best
results, but it's about 10x faster than GPT-4, and also cheaper to use. You can change the model used
by setting the `fable_saga.default_openai_model_name` parameter to the model of your choice. You can also go with another model
provider supported by LangChain and pass that in when creating the agent.

The demo uses OpenAI because it's the easiest to get started with, is very effective, and supports json output well.
SAGA parses the output as JSON, so you can use any model that outputs JSON. If you use a different model, you may need
to add more specifics to the context that the model should only generate valid JSON or fine tune the model to
produce better results.

Installing as a PyPi Package
-------------
This will be coming once development settles a little first.


Installing Dependencies
==================

Python on Windows
----
Install Python 3.11.4
* Download and Install [Python 3.11.4](https://www.python.org/downloads/release/python-3114/) (earlier Python 3 versions may also work)
* Add python to your **Path** environment variable (e.g. `C:\Python\Python311\python.exe`)

Python on OSX and Linux
---
This is generally easier but platform specific. We recommend using homebrew on OSX and your integrated manager on linux (e.g. apt-get).


Install Poetry and Dependencies
-----
* Download and install the latest [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
* Add poetry to your **Path** environment variable (e.g. `C:\Users\Username\AppData\Roaming\pypoetry\venv\Scripts`)
* Run this command in the fable-generative-agents repo: `poetry install`

Configure OpenAI
---------
In order to avoid having to pass the OPENAI_API_KEY to the demo every time, you can set it as an environment variable.
* WINDOWS: Create a System environment variable called **OPENAI_API_KEY** and paste your API key into the value field
* ELSEWHERE: Creating a .env file or adding it to your bash profile should work. Many IDEs also have a way to set environment variables.

Install PyCharm (optional)
--------
Just a recommendation for our internal team who want to get this setup. Community folks can use any IDE you want.
* Download and install the latest [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/other.html)
* Open fable-generative-agents in PyCharm
* Configure interpreter: 
  - Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter > Poetry Environment
  - Base interpreter: `C:\Users\<Username>\AppData\Roaming\pypoetry\venv\Scripts\python.exe`
  - Check Install packages from pyproject.toml
  - Poetry executable: `C:\Users\<Username>\AppData\Roaming\Python\Scripts\poetry.exe`
  - Choose 'OK'



