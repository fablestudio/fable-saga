import json
import pathlib
from typing import List, Optional, Dict, Any

import cattrs
from attr import define
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import load_prompt

from . import (
    BaseSagaAgent,
)


@define(slots=True)
class ConversationTurn:
    persona_guid: str
    dialogue: str


@define(slots=True)
class GeneratedConversation:
    conversation: List[ConversationTurn]
    raw_prompt: Optional[str] = None
    raw_response: Optional[str] = None
    llm_info: Optional[Dict[str, Any]] = None
    retries: int = 0
    error: Optional[str] = None


class ConversationAgent(BaseSagaAgent):
    """Agent that generates conversation from a context and a list of personas."""

    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """Initialize the conversation agent.

        Args:
            llm: The language model to use for generation. Defaults to OpenAI. (see LangChain docs).
        """
        super().__init__(
            prompt_template=load_prompt(
                pathlib.Path(__file__).parent.resolve()
                / "prompt_templates/generate_conversation.yaml"
            ),
            llm=llm,
        )

    async def generate_conversation(
        self,
        persona_guids: List[str],
        context: str,
        max_tries=0,
        verbose=False,
        model_override: Optional[str] = None,
    ) -> GeneratedConversation:
        """Generate conversation for the given personas and context

        Args:
            persona_guids: A list of persona guids to generate conversation for. They probably match information in the
                context.
            context: The context to generate conversation for like character descriptions, setting, etc.
            max_tries: The maximum number of retries to attempt if no options are found.
            verbose: Whether to print verbose output.
            model_override: The model to use for generation (allows for switching llm.model_name at runtime).
        """
        assert (
            persona_guids is not None and len(persona_guids) > 0
        ), "Must provide at least one persona_guid."
        assert context is not None and len(context) > 0, "Must provide a context."

        chain = self.generate_chain(model_override)
        chain.verbose = verbose

        retries = 0
        last_error = None
        formatted_persona_guids = "[" + ", ".join(persona_guids) + "]"

        while retries <= max_tries:
            try:
                response = await chain.ainvoke(
                    {"context": context, "persona_guids": formatted_persona_guids},
                    {"callbacks": [self.callback_handler]},
                )
                raw_response = response.get("text")
                if raw_response is None:
                    raise Exception("No text key found in response.")
                if len(raw_response) == 0:
                    raise Exception("Text is empty in response.")
                if not isinstance(raw_response, str):
                    raise Exception(
                        f"Text is not a string in response but {type(raw_response)}."
                    )

                raw_conversation = json.loads(raw_response)
                # If we parse the results, but didn't get any options, retry. Should be rare.
                if raw_conversation.get("conversation") is None:
                    raise Exception("No conversation key found in JSON response.")
                if len(raw_conversation["conversation"]) == 0:
                    raise Exception("conversation list is empty in JSON response.")

                # Convert the options to a GeneratedConversation object and add metadata.
                conversation = cattrs.structure(raw_conversation, GeneratedConversation)
                conversation.raw_prompt = self.callback_handler.last_prompt
                conversation.raw_response = raw_response
                conversation.llm_info = self.callback_handler.last_model_info
                conversation.retries = retries
                return conversation

            except (json.JSONDecodeError, TypeError) as e:
                last_error = f"Error decoding response: {str(e)}"
            except cattrs.errors.ClassValidationError as e:
                last_error = f"Error validating response: {cattrs.transform_error(e)}"
            except Exception as e:
                last_error = str(e)
            retries += 1

        return GeneratedConversation(
            conversation=[],
            retries=retries,
            raw_response=self.callback_handler.last_generation,
            raw_prompt=self.callback_handler.last_prompt,
            llm_info=self.callback_handler.last_model_info,
            error=f"No options found after {retries} retries."
            f" Last error: {last_error}",
        )
