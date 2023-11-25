import json
import random
from datetime import datetime
from typing import List, Callable, Optional, Any, Dict

from cattr import unstructure

from fable_agents import ai, models
from fable_agents.ai import Agent
from models import Persona, Vector3, StatusUpdate, Message, ObservationEvent, SequenceUpdate, MetaAffordanceProvider, \
    Conversation, Location, LocationNode, EntityId
from fable_agents.datastore import Datastore, MetaAffordances
import socketio

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import load_prompt


sio: socketio.AsyncServer = None
simulation_client_id: Optional[str] = None

debug=True


class Format:
    @staticmethod
    def persona(persona: Persona):
        return {
            'persona_guid': persona.guid,
            'first_name': persona.first_name,
            'last_name': persona.last_name,
            'description': persona.description,
            'summary': persona.summary,
            'backstory': persona.backstory
        }

    @staticmethod
    def persona_short(persona):
        return {
            'persona_guid': persona.guid,
            'name': persona.first_name + "" + persona.last_name,
            'summary': persona.summary,
        }

    @staticmethod
    def observation_event(event: ObservationEvent) -> Dict[str, str]:
        out = {
            'persona_guid': event.persona_guid,
            'action': event.action,
            'action_step': event.action_step,
            'distance': str(round(event.distance, 2)) + 'm',
        }
        # If the event has a summary, then we don't need to include the action and action_step.
        if event.summary is not None and event.summary != '':
            del out['action'], out['action_step']
            out['summary'] = event.summary
        # If the event has an importance, only show it if it's greater than 0.
        if event.importance is not None and type(event.importance) == int and event.importance > 0:
            out['importance'] = event.importance
        return out

    @staticmethod
    def observer(status_update: StatusUpdate):
        out = {
            'action': status_update.sequence,
            'action_step': status_update.sequence_step,
        }
        if status_update.position is not None and status_update.destination_id is not None:
            destination = Datastore.locations.locations.get(status_update.destination_id, None)
            if destination is None:
                print('Error: location not found: ', status_update.destination_id)
            else:
                out['destination_distance'] = str(float(Vector3.distance(
                    status_update.position, destination.center_floor_position))) + "m",
        return out

    @staticmethod
    def simple_datetime(dt: datetime):
        # Format the datetime as a string with the format: Monday, 12:00 PM
        return dt.strftime("%A, %I:%M %p")

    @staticmethod
    def simple_time_ago(dt: datetime, current_datetime: datetime) -> str:
        # Format as the number of days or minutes ago. Only use days if it was at least a day ago.
        days_ago = (current_datetime - dt).days
        minutes_ago = int((current_datetime - dt).total_seconds() / 60)
        if days_ago > 0:
            return str(days_ago) + ' days ago'
        else:
            return str(minutes_ago) + 'm ago'

    @staticmethod
    def sequence_updates(sequence_updates: List[SequenceUpdate], timestamp: datetime):
        output = []
        last_action = None
        for sequence_update in sequence_updates:
            if sequence_update.action != last_action:
                output.append({
                    'started': Format.simple_time_ago(sequence_update.timestamp, timestamp),
                    'last_time': Format.simple_time_ago(sequence_update.timestamp, timestamp),
                    'action': sequence_update.action,
                    'steps': sequence_update.action_step_started
                })
                last_action = sequence_update.action
            else:
                output[-1]['steps'] += " > " + sequence_update.action_step_started
                output[-1]['last_time'] = Format.simple_time_ago(sequence_update.timestamp, timestamp)
        return output

    @staticmethod
    def interaction_option(provider: MetaAffordanceProvider):
        return {
            'item_guid': provider.sim_object.guid,
            'name': provider.sim_object.display_name,
            'description': provider.sim_object.description,
            'interactions': provider.affordances
        }

    @staticmethod
    def conversation(conversation: Conversation, current_datetime: datetime):

        turns = []
        for turn in conversation.turns:
            turns.append({
                turn.guid: turn.dialogue,
            })

        return {
            'time_ago': str(int((current_datetime - conversation.timestamp).total_seconds() / 60)) + 'm ago',
            'transcript': turns,
        }

    @staticmethod
    def location_tree(nodes: List[LocationNode]) -> Dict[str, Any]:
        output = {}

        # Recursively generate a tree of locations.
        def gen_tree(node: LocationNode):
            return {
                'name': node.location.name,
                'description': node.location.description,
                'children': [gen_tree(child) for child in node.children]
            }

        for node in nodes:
            # Only add the root nodes. The rest will be included in the tree.
            if node.parent is None:
                output[node.location.name] = gen_tree(node)
        return output

    @staticmethod
    def memories(memories: List[models.Memory], current_datetime: datetime) -> List[Dict[str, str]]:
        output = []
        for memory in memories:
            output.append({
                'time_ago': Format.simple_time_ago(memory.timestamp, current_datetime),
                'summary': memory.summary,
                # TODO: Add more details like the positions and entity ids perhaps.
            })
        return output


class Resolution:
    HIGH = 'HIGH'
    MEDIUM = 'MEDIUM'
    LOW = 'LOW'


class GaiaAPI:

    observation_distance = 10
    observation_limit = 10

    async def create_conversation(self, persona_guid:str, on_complete: Callable[[Message], None]):
        """
        Create a new conversation.
        :param persona_guids: unique identifiers for the personas.
        :param callback: function to call when the conversation is created.
        """
        initiator_persona = Datastore.personas.personas.get(persona_guid, None)
        if initiator_persona is None:
            print('Error: persona not found.')
            if on_complete is not None:
                on_complete(Message('error', {'error': 'persona not found.'}))
            return

        # Create a list of personas to send to the server as options.
        options = []

        # TODO: Using all personas results in too much text.
        # pick 5 random personas
        items = list(Datastore.personas.personas.items())
        random.shuffle(items)
        for guid, persona_option in items[:5]:
            options.append(Format.persona(persona_option))

        prompt = load_prompt("prompt_templates/conversation_v1.yaml")
        llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")
        chain = LLMChain(llm=llm, prompt=prompt)
        resp = await chain.arun(self_description=json.dumps(Format.persona(initiator_persona)), options=json.dumps(options))
        if on_complete is not None:
            on_complete(resp)

    async def create_observations(self, observer_update: StatusUpdate, updates_to_consider: List[StatusUpdate]) -> List[Dict[str, Any]]:
        """
        Create a new observation.
        :param persona_guids: unique identifiers for the personas.
        :param callback: function to call when the conversation is created.
        """
        initiator_persona = Datastore.personas.personas.get(observer_update.guid, None)
        if initiator_persona is None:
            print('Error: persona not found.')
            return []

        # Create a list of personas to send to the server as options.
        observation_events: Dict[str, ObservationEvent] = {}

        # sort by distance for now. Later we might include a priority metric as well.
        updates_to_consider.sort(key=lambda x: Vector3.distance(x.position, observer_update.position))
        for update in updates_to_consider[:self.observation_limit]:
            # Ignore the observer when creating observations.
            if update.guid == observer_update.guid:
                continue

            if update.position is not None and Vector3.distance(update.position, observer_update.position) <= self.observation_distance:
                observation_events[update.guid] = ObservationEvent.from_status_update(update, observer_update)

        #### Don't use the LLM for now. It's too slow if we're using it for every character.
        # prompt = load_prompt("prompt_templates/observation_v1.yaml")
        # llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")
        # chain = LLMChain(llm=llm, prompt=prompt)
        # #pprint(observation_events)
        # formatted_obs_events = json.dumps([Format.observation_event(evt) for evt in observation_events.values()])
        # #print("create_observations:llm OBS::", formatted_obs_events)
        # resp = await chain.arun(time=Format.simple_datetime(observer_update.timestamp),
        #                         self_description=json.dumps(Format.persona(initiator_persona)),
        #                         self_update=json.dumps(Format.observer(observer_update)),
        #                         update_options=formatted_obs_events)
        #
        # # Create observations for the observer.
        # intelligent_observations: [Dict[str, Any]] = []
        # #print("create_observations:llm RESP::", resp)
        # if resp:
        #     try:
        #         intelligent_observations = json.loads(resp)
        #     except json.decoder.JSONDecodeError as e:
        #         print("Error decoding response", e, resp)
        #         intelligent_observations = []
        #
        #     for observation in intelligent_observations:
        #         guid = observation.get('guid', None)
        #         if guid in Datastore.personas.personas.keys() and observation_events.get(guid, None):
        #             event = observation_events[guid]
        #             event.summary = observation.get('summary_of_activity', '')
        #             event.importance = observation.get('summary_of_activity', 0)
        #             Datastore.memory_vectors.memory_vectors.save_context({'observation_event': event},{'summary_of_activity': event.summary, 'importance': event.importance})
        Datastore.observation_memory.set_observations(initiator_persona.guid, observer_update.timestamp,
                                                      list(observation_events.values()))
        # return intelligent_observations

    async def create_reactions(self, resolution: str, persona_id: str, observer_update: StatusUpdate, observations: List[ObservationEvent],
                               sequences: [List[SequenceUpdate]], metaaffordances: MetaAffordances,
                               conversations: List[Conversation], personas: List[Persona],
                               recent_goals: List[str], current_timestamp: datetime,
                               default_action: str) -> List[Dict[str, Any]]:

        initiator_persona = Datastore.personas.personas.get(persona_id, None)
        if initiator_persona is None:
            print('Error: persona not found.')
            return []

        # memories = datastore.memory_vectors.load_memory_variables({'context': context})

        # Create a list of actions to consider.
        action_options = ai.Actions.copy()

        # TODO: Perhaps refine the number of actions generated or other options based on the resolution.
        if resolution == Resolution.HIGH:
            prompt = load_prompt("prompt_templates/actions_v1.yaml")
            # For now we load the latest GPT-4 for high resolution.
            model_name = "gpt-4-1106-preview"
            retries = 1
        else:
            # For now we load the latest GPT-3 for all others.
            prompt = load_prompt("prompt_templates/actions_v1.yaml")
            model_name = "gpt-3.5-turbo-1106"
            # Remove converse_with for right now for low resolution.
            action_options = [item for item in action_options if item.get('action') != 'converse_with']
            # If it fails, don't retry.
            retries = 0

        # Append the default action to the list of options if one is provided.
        if default_action is not None and default_action != '':
            action_options.append({'action': 'default_action',
                                   'description': "Do what you would normally do in this context: " + default_action, 'parameters': {}})

        llm = ChatOpenAI(temperature=0.9, model_name=model_name, model_kwargs={
            "response_format": {"type": "json_object"}})
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        options = []
        while retries >= 0 and len(options) == 0:
            resp = await chain.arun(time=Format.simple_datetime(current_timestamp),
                                    persona_guid=persona_id,
                                    self_description=json.dumps(Format.persona(initiator_persona)),
                                    self_update=json.dumps(Format.observer(observer_update)),
                                    observations=json.dumps([Format.observation_event(evt) for evt in observations]),
                                    sequences=json.dumps(Format.sequence_updates(sequences, current_timestamp)),
                                    action_options=json.dumps(action_options),
                                    conversations=json.dumps([Format.conversation(convo, current_timestamp) for convo in conversations]),
                                    personas=json.dumps([Format.persona_short(persona) for persona in personas]),
                                    interact_options=json.dumps(
                                        [Format.interaction_option(affordance) for affordance in metaaffordances.affordances.values()]),
                                    recent_goals=json.dumps(recent_goals),
                                    locations=json.dumps(Format.location_tree(list(Datastore.locations.nodes.values()))),
                                    memories=json.dumps(Format.memories(Datastore.memories.get(persona_id), current_timestamp)),
                                    )

            try:
                options = json.loads(resp)
            except (json.JSONDecodeError, TypeError) as e:
                print("Error decoding response", e, resp)
                options = []
            if len(options) == 0:
                print("No options found. Retrying.")
                retries -= 1
            else:
                break
        return options


class SimulationAPI:

    def get_agent(self, guid: str):
        return Datastore.agents.get(guid)

    async def reload_personas(self, guids: List[str], on_complete: Optional[Callable[[], None]]):
        """
        Load persona's current state (i.e. description and memory).
        :param on_complete:
        :param guids: unique identifiers for the persona. If empty, load all
        :param callback: function to call when the personas are loaded.
        """

        def convert_to_personas(response: Message):
            #print("RESPONSE", response)
            if response.type != 'request-personas-response':
                print('Error: expected personas response.', response)
                if on_complete is not None:
                    on_complete()
                return
            for json_rep in response.data['personas']:
                persona = Persona.from_json(json_rep)
                Datastore.personas.personas[persona.guid] = persona

            # TODO: This should return when this process is complete.
            if on_complete is not None:
                on_complete()

        # Note: Server callback only works with a specific client id.
        await self.send('request-personas', {'guids': guids}, callback=convert_to_personas)

    async def reload_affordances(self, guids: List[str], on_complete: Optional[Callable[[], None]]):
        """
        Load all affordances from the active scene
        :param on_complete:
        :param guids: unique identifiers to load. If empty, load all
        :param callback: function to call when the affordances are loaded.
        """

        def convert_to_affordances(response: Message):
            if response.type != 'request-affordances-response':
                print('Error: expected affordances response.', response)
                if on_complete is not None:
                    on_complete()
                return
            # Load meta affordances
            Datastore.meta_affordances.affordances.clear()
            for json_rep in response.data['meta_affordances']:
                provider = MetaAffordanceProvider.from_json(json_rep)
                Datastore.meta_affordances.affordances[provider.sim_object.guid] = provider

            # TODO: Load generic NPC affordances and all other sim object affordances

            # TODO: This should return when this process is complete.
            if on_complete is not None:
                on_complete()

        # Note: Server callback only works with a specific client id.
        await self.send('request-affordances', {'guids': guids}, callback=convert_to_affordances)

    async def reload_locations(self, on_complete: Optional[Callable[[], None]]):
        """
        Load all affordances from the active scene
        :param on_complete:
        :param callback: function to call when the affordances are loaded.
        """

        def convert_to_locations(response: Message):
            if response.type != 'request-locations-response':
                print('Error: expected locations response.', response)
                if on_complete is not None:
                    on_complete()
                return
            for json_rep in response.data['locations']:
                location = Location.from_json(json_rep)
                Datastore.locations.locations[location.guid] = location

            # TODO: This should return when this process is complete.
            if on_complete is not None:
                on_complete()

        # Note: Server callback only works with a specific client id.
        await self.send('request-locations', {}, callback=convert_to_locations)

    async def send(self, type: str, data: dict, callback: Callable[[Message], None]):
        """
        Send a request to the server.
        :param type: type of request.
        :param data: data to send with the request.
        :param callback: function to call when the server responds.
        """

        def convert_response_to_message(response_type: str, response_data: str):
            response_message = Message(response_type, json.loads(response_data))
            callback(response_message)

        if simulation_client_id is None:
            print('Error: simulation_client_id is None.')
            if callback is not None:
                callback(Message('error', {'error': 'simulation_client_id is None.'}))
            return
        if callback is not None:
            if debug: print('emit', 'message-ack', type, json.dumps(data), "to=" + simulation_client_id,
                            "callback=" + str(callback))
            await sio.emit('message-ack', (type, json.dumps(data)), to=simulation_client_id,
                           callback=convert_response_to_message)

        else:
            if debug: print('emit', 'message', type, json.dumps(data))
            await sio.emit('message', (type, json.dumps(data)))


class Agents:

    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def get(self, guid: EntityId, create=True) -> Agent:
        found = self.agents.get(guid, None)
        if not found and create:
            found = self.agents[guid] = Agent(guid)
        return found


class API:
    simulation: SimulationAPI = SimulationAPI()
    gaia: GaiaAPI = GaiaAPI()
    agents: Agents = Agents()
