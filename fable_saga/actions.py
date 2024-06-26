import json
import pathlib
from typing import Any, Dict, List, Optional

import cattrs
from attr import define
from cattr import unstructure, structure
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import load_prompt

from . import BaseSagaAgent


@define(slots=True)
class Skill:
    name: str
    description: str
    parameters: Dict[str, str] = {}


@define(slots=True)
class Action:
    skill: str
    parameters: Dict[str, Any] = {}


@define(slots=True)
class GeneratedActions:
    options: List[Action]
    scores: List[float]
    raw_prompt: Optional[str] = None
    raw_response: Optional[str] = None
    llm_info: Optional[Dict[str, Any]] = None
    retries: int = 0
    error: Optional[str] = None

    def sort(self):
        """Sort the actions by score."""
        self.options = [
            x
            for _, x in sorted(
                zip(self.scores, self.options), key=lambda pair: pair[0], reverse=True
            )
        ]
        self.scores = sorted(self.scores, reverse=True)


class ActionsAgent(BaseSagaAgent):
    """Agent that generates actions from a context and a list of skills."""

    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__(
            prompt_template=load_prompt(
                pathlib.Path(__file__).parent.resolve()
                / "prompt_templates/generate_actions.yaml"
            ),
            llm=llm,
        )

        self.guidance = """
* Provide a goal for your character to achieve as well as the action.
* Do not make up new characters.
* Staying true to your character's personality and circumstances.
* Use your observations and conversations to inform your actions.
* Only use provided skills to create action options (do not make up skills).
* Do not do the same thing again if you have already done it.
* Advance the story by generating actions that will help you achieve your most important and immediate goals.
"""

    async def generate_actions(
        self,
        context: str,
        skills: List[Skill],
        max_tries=0,
        verbose=False,
    ) -> GeneratedActions:
        """Generate actions for the given context and skills.

        Args:
            context: The context for the action generation, this is added to the prompt along with the
                hardcoded guidance.
            skills: The list of skills to use for action generation, this is added to the prompt.
            max_tries: The maximum number of tries to generate actions.
            verbose: Whether to print verbose output during generation.
        """
        assert context is not None and len(context) > 0, "Must provide a context."
        assert (
            skills is not None and len(skills) > 0
        ), "Must provide at least one skill."
        for skill in skills:
            assert isinstance(skill, Skill), "Must provide a list of Skill objects."
            assert (
                skill.name is not None and len(skill.name) > 0
            ), "Must provide a skill name."
            assert (
                skill.description is not None and len(skill.description) > 0
            ), "Must provide a skill description."

        chain = self.generate_chain()
        chain.verbose = verbose

        retries = 0
        last_error = None
        dumped_skills = [unstructure(skill) for skill in skills]
        json_skills = json.dumps(dumped_skills)

        # Try to generate actions up to max_tries times. Sometimes the model doesn't return any options or the
        # response is invalid JSON.
        while retries <= max_tries:
            raw_response = None
            try:
                response: dict = await chain.ainvoke(
                    {"context": context, "skills": json_skills},
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

                raw_actions = json.loads(raw_response)
                # If we parse the results, but didn't get any options, retry. Should be rare.
                if raw_actions.get("options") is None:
                    raise Exception("No options key found in JSON response.")
                if len(raw_actions["options"]) == 0:
                    raise Exception("options list is empty in JSON response.")

                # Convert the options to a GeneratedActions object and add metadata.
                actions = structure(raw_actions, GeneratedActions)
                actions.raw_prompt = self.callback_handler.last_prompt
                actions.raw_response = raw_actions
                actions.llm_info = self.callback_handler.last_model_info
                actions.retries = retries

                # Sort the actions by score to be helpful before returning.
                actions.sort()
                return actions
            except (json.JSONDecodeError, TypeError) as e:
                last_error = f"Error decoding response: {str(e)}"
            except cattrs.errors.ClassValidationError as e:
                last_error = f"Error validating response: {cattrs.transform_error(e)}"
            except Exception as e:
                last_error = str(e)
            finally:
                # If there was an error, add the raw response to the error message.
                if last_error:
                    last_error += f"Raw response: {raw_response}"
            retries += 1
        return GeneratedActions(
            options=[],
            scores=[],
            retries=retries,
            raw_response=self.callback_handler.last_generation,
            raw_prompt=self.callback_handler.last_prompt,
            llm_info=self.callback_handler.last_model_info,
            error=f"No options found after {retries} retries."
            f" Last error: {last_error}",
        )
