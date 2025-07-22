import logging
import json
import os
from textwrap import dedent
from typing import Any, List, Optional, Union, Dict
from crew import Crew
from tool import Tool
from utils import create_message, MessageHistory, TagParser
from dotenv import load_dotenv
import litellm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

REACT_PROMPT = """
1. Your worflow is by running a ReAct (Reasoning + Action) loop with the following steps: 
    - Thought
    - Action
    - Observation
2. You have access to function signatures within <tools></tools> tags.
3. Feel free to invoke one or more of these functions to address the user's request. Do not presume default values—always use the signatures provided.
4. If a tool/function is available for some task you MUST use it, without fail.
5. You are someone who only knows how to write, for any sort of data, see what tools are available and how to use them.
6. Pay close attention to each function's types property and supply arguments exactly as a Python dictionary.
7. After successful tool call, give <observation> result from tool call </observation> and <response> final output </response>.
8. For every function call, return a JSON object wrapped in <tool_call></tool_call> tags, using this pattern:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <sequential-id>}
</tool_call>

Here are the available tools/actions/functions to assist you:

<tools>
%s
</tools>


Example:

<question>User Question</question>
<thought> Some Thought </thought>
<tool_call>{"name":"abc","arguments":{"pqr":"xyz"},"id":0}</tool_call>
<observation>{0: {"result":25}}</observation>
<response>The result is 25</response>

Note: If the user's question doesn't require a tool, then DO NOT use tool and answer directly inside <response> tags.
"""

class Agent:
    """
    Unified Agent supporting both standard LLM and ReAct (tool-using) operation.
    Now also directly handles LLM API setup and response generation (formerly ChatClientWrapper).
    """
    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str,
        expected_output_format: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        llm_model: str = "gpt-4o",
    ) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise RuntimeError(
                "OPENAI_API_KEY must be set to use OpenAI via LiteLLM")
        litellm.api_key = api_key
        self.model = f"openai/{llm_model}"

        self.name: str = name
        self.backstory: str = backstory
        self.task_description: str = task_description
        self.expected_output_format: Optional[str] = expected_output_format
        self.tools: List[Tool] = tools or []
        self.llm_model: str = llm_model

        # Tool registry for ReAct
        self.tools_dict: Dict[str, Tool] = {t.signature.name: t for t in self.tools}
        self._parse_response = TagParser("response")
        self._parse_thought = TagParser("thought")
        self._parse_tool = TagParser("tool_call")

        self.dependencies: List["Agent"] = []
        self.dependents: List["Agent"] = []
        self.context_messages: List[str] = []

        Crew.register(self)
        logger.debug("Agent %s initialized and registered", self.name)

    def _generate_response(self, conversation: List[Dict[str, str]]) -> str:
        """
        Uses LiteLLM to generate a response from OpenAI models.
        """
        try:
            response = litellm.completion(
                model=self.model,
                messages=conversation
            )
            return response["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("Error calling OpenAI via LiteLLM")
            raise

    def __repr__(self) -> str:
        return f"<Agent {self.name!r}>"

    def precedes(self, other: "Agent") -> "Agent":
        """
        Use `agent_a.precedes(agent_b)` to make A a dependency of B (A runs before B).
        This will wire the dependency and dependent relationship directly.
        """
        if not isinstance(other, Agent):
            raise TypeError("Argument must be an Agent instance")
        self.dependents.append(other)
        other.dependencies.append(self)
        logger.debug("%s now precedes %s (is a dependency of)", self.name, other.name)
        return other

    def succeeds(self, other: "Agent") -> "Agent":
        """
        Use `agent_b.succeeds(agent_a)` to make A a dependency of B (A runs before B).
        This will wire the dependency and dependent relationship directly.
        """
        if not isinstance(other, Agent):
            raise TypeError("Argument must be an Agent instance")
        self.dependencies.append(other)
        other.dependents.append(self)
        logger.debug("%s now succeeds %s (depends on)", self.name, other.name)
        return other

    def receive_context(self, data: Any) -> None:
        """
        Receive and store context data from another agent.

        Args:
            data: Output produced by a dependency.
        """
        message = f"From {self.name}'s dependency: {data}"
        self.context_messages.append(message)
        logger.debug("%s received context: %s", self.name, data)

    def _build_prompt(self) -> str:
        context_block = "\n".join(self.context_messages)
        prompt = dedent(
            f"""
            <task_description>
            {self.task_description}
            </task_description>

            <task_expected_output>
            {self.expected_output_format or ""}
            </task_expected_output>

            <context>
            {context_block}
            </context>
            """
        ).strip()
        return prompt

    def _react_prompt(self) -> str:
        tools_block = "\n".join(t.info() for t in self.tools)
        return REACT_PROMPT % tools_block

    def _run_tool_calls(self, calls: List[str]) -> Dict[int, Any]:
        observations: Dict[int, Any] = {}
        for call_str in calls:
            call = json.loads(call_str)
            name = call["name"]
            tool = self.tools_dict.get(name)
            if tool is None:
                raise KeyError(f"Tool '{name}' not found in registry")
            args = call["arguments"]
            print("-"*50)
            print(f"{'-'*50}\nInvoking tool {name} with {args}\n{'-'*50}\n")
            result = tool(**args)
            print(f"\n{'-'*50}\nResult from {name}: {result}\n{'-'*50}\n")
            print("-"*50)
            observations[int(call["id"])] = result
        return observations

    def run(self, user_message: Optional[str] = None, max_rounds: int = 10) -> Any:
        """
        Execute this agent's task. If tools are provided, use ReAct loop; otherwise, standard LLM.
        """
        prompt = self._build_prompt() if user_message is None else user_message
        logger.info("Agent %s running with prompt: %s", self.name, prompt)

        if self.tools:
            # ReAct loop
            history = MessageHistory()
            system_prompt = self.backstory + "\n" + self._react_prompt()
            history.append(create_message("system", system_prompt))
            history.append(create_message("user", prompt, tag="question"))
            for round_idx in range(max_rounds):
                logging.debug("Round %d history: %s", round_idx, history.all())
                completion = self._generate_response(history.all())
                resp = self._parse_response.parse(completion)
                if resp.found:
                    result = resp.items[0]
                    break
                thought = self._parse_thought.parse(completion)
                tool_calls = self._parse_tool.parse(completion)
                history.append(create_message("assistant", completion))
                if thought.found:
                    logging.debug("Agent thought: %s", thought.items[0])
                if tool_calls.found:
                    observations = self._run_tool_calls(tool_calls.items)
                    logging.debug("Observations: %s", observations)
                    history.append(create_message("user", json.dumps(observations)))
                    continue
            else:
                # Fallback
                logging.warning("Max rounds reached; returning fallback response")
                result = self._generate_response(history.all())
        else:
            # Standard LLM agent
            system_prompt = self.backstory
            history = MessageHistory()
            history.append(create_message("system", system_prompt))
            history.append(create_message("user", prompt))
            result = self._generate_response(history.all())

        logger.info("Agent %s produced result: %s", self.name, result)
        for dep in self.dependents:
            dep.receive_context(result)
        return result
