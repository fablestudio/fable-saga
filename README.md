

Demo Quickstart
-------------
1. Make sure you have git, python-3.10/11, and poetry installed and OPENAI key setup (see below).
2. Clone this repo and change the directory to it.
3. run `poetry install` to install dependencies.
4. start a poetry shell with `poetry shell` to make sure you are using the correct python version and have the correct environment variables set.
5. run `OPENAI_API_KEY=<YOUR KEY> python demos/space_colony/simulation.py`

B.Y.O.S. (Bring Your Own Simulation)
--------
SAGA is a library that can be used to generate actions for any simulation. Our blog post shows
SAGA being used in our "Thistle Gulch" simulation, which is full on 3D interactive simulation, but to
get started, you can use the Space Colony demo as a starting point. External simulations (like Thistle Gulch)
can connect to SAGA via socketio. We removed that code from the demo to make it easier to get started, but
we will be adding it back in soon.

Space Colony Demo
==================

The Space Colony demo is a simple text-based simulation of a spaceship with a crew of 5 agents.
There are various locations on the ship that the agents can move to and objects they can interact
with. All of this data we call "Meta-Memories" and they formatted and provided to the `fable_saga.Agent`
when requesting actions.

Generating Actions from Skills
-------------
Right now the agents and their spaceship is very simple, but with the few actions they have, they
can generate a lot of interesting behavior. The actions they can take are defined by "Skills" which
are listed in the `demos/space_colony/resources/skills.yaml` file. Here are two of them:

```Yaml
- guid: go_to
  description: "Go to a location in the world"
  parameters:
    destination: "<str: persona_guid, item_guid, or location.name to go to>"
    goal: "<str: goal of the movement>"

- guid: converse_with
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
You can also use the `fable_saga.Agent` class outside the demo to generate actions for your own sim. 
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
the `fable_saga.Agent` class. The demo simulation keeps track of these memories, not SAGA.

The memories will continue to be generated as the agents interact with the world. This can add a lot of
tokens to the context data, so if you have any intention of running the demo for a long time, you will
want to implement some kind of memory management system to keep the context data from getting too large.

Large Language Models
-------------
The demo uses the `gpt-3.5-turbo-1106` model from OpenAI by default. This model doesn't produce the best
results, but it's about 10x faster than GPT-4, and also cheaper to use. You can change the model used
by setting the `fable_saga.default_openai_model_name` parameter to the model of your choice. You can also go with another model
provider supported by LangChain. The demo uses OpenAI because it's the easiest to get started with.

SAGA parses the output as JSON, so you can use any model that outputs JSON.

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



