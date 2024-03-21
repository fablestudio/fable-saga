import json
import pathlib
from typing import List, Optional, Dict, Any

import cattrs
from attr import define
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import load_prompt

from . import (
    default_openai_model_name,
    default_openai_model_temperature,
    logger,
    SagaCallbackHandler,
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


class ConversationAgent:

    def __init__(self, llm: BaseChatModel = None):
        self._llm = (
            llm
            if llm is not None
            else ChatOpenAI(
                temperature=default_openai_model_temperature,
                model_name=default_openai_model_name,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        )
        path = pathlib.Path(__file__).parent.resolve()
        self.generate_conversation_prompt = load_prompt(
            path / "prompt_templates/generate_conversation.yaml"
        )

    def generate_chain(self, model_override: Optional[str] = None) -> LLMChain:
        self._llm.model_name = (
            model_override if model_override else default_openai_model_name
        )
        return LLMChain(llm=self._llm, prompt=self.generate_conversation_prompt)

    async def generate_conversation(
        self,
        persona_guids: List[str],
        context: str,
        max_tries=0,
        verbose=False,
        model_override: Optional[str] = None,
    ):
        """Generate conversation for the given personas and context"""
        assert (
            persona_guids is not None and len(persona_guids) > 0
        ), "Must provide at least one persona_guid."
        assert context is not None and len(context) > 0, "Must provide a context."

        chain = self.generate_chain(model_override)
        chain.verbose = verbose

        # Set up the callback handler.
        callback_handler = SagaCallbackHandler(
            prompt_callback=lambda prompts: logger.info(f"Prompts: {prompts}"),
            response_callback=lambda result: logger.info(f"Response: {result}"),
        )

        retries = 0
        last_error = None
        formatted_persona_guids = "[" + ", ".join(persona_guids) + "]"

        while retries <= max_tries:
            try:
                last_response = await chain.arun(
                    context=context,
                    persona_guids=formatted_persona_guids,
                    callbacks=[callback_handler],
                )
                raw_conversation = json.loads(last_response)
                # If we parse the results, but didn't get any options, retry. Should be rare.
                if raw_conversation.get("conversation") is None:
                    raise Exception("No conversation key found in JSON response.")
                if len(raw_conversation["conversation"]) == 0:
                    raise Exception("conversation list is empty in JSON response.")

                # Convert the options to a GeneratedConversation object and add metadata.
                conversation = cattrs.structure(raw_conversation, GeneratedConversation)
                conversation.raw_prompt = callback_handler.last_prompt
                conversation.raw_response = last_response
                conversation.llm_info = callback_handler.last_model_info
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
            raw_response=callback_handler.last_generation,
            raw_prompt=callback_handler.last_prompt,
            llm_info=callback_handler.last_model_info,
            error=f"No options found after {retries} retries."
            f" Last error: {last_error}",
        )
