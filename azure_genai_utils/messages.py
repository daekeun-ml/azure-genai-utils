import uuid
from langchain_core.messages import AIMessageChunk
from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass
from langchain_core.agents import AgentAction, AgentStep
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


def random_uuid():
    return str(uuid.uuid4())


def stream_response(response, return_output=False):
    """
    Streams the response from the AI ​​model, processing each chunk as it is output.

    This function iterates over each item in the `response` iterable. If the item is an instance of `AIMessageChunk`,
    it extracts the contents of the chunk and outputs it. If the item is a string, it outputs the string directly. Optionally, the function
    can return a concatenated string of all the response chunks.

    :param response (iterable): An iterable of response chunks that can be `AIMessageChunk` objects or strings.
    :param return_output (bool, optional): If True, the function returns the concatenated response string. Default is False.
    :return str: The concatenated response string if `return_output` is True. Otherwise, nothing is returned.
    """
    answer = ""
    for token in response:
        if isinstance(token, AIMessageChunk):
            if isinstance(token.content, list):
                answer += "".join(str(item) for item in token.content)
            else:
                answer += token.content
            print(token.content, end="", flush=True)
        elif isinstance(token, str):
            answer += token
            print(token, end="", flush=True)
    if return_output:
        return answer


def tool_callback(tool) -> None:
    """
    Callback function for tool usage.
    """
    print("Toll Usage]")
    print(f"Tool: {tool.get('tool')}")
    if tool_input := tool.get("tool_input"):
        for k, v in tool_input.items():
            print(f"{k}: {v}")
    print(f"Log: {tool.get('log')}")


def observation_callback(observation) -> None:
    """
    Callback function for observation.
    """
    print("[Observation]")
    print(f"Observation: {observation.get('observation')}")


def result_callback(result: str) -> None:
    """
    Callback function for the final result.
    """
    print("[Final Result]")
    print(result)


@dataclass
class AgentCallbacks:
    """
    Data class containing agent callback functions.
    :param tool_callback: Callback function for tool usage.
    :param observation_callback: Callback function for observation.
    :param result_callback: Callback function for the final result.
    """

    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback


class AgentStreamParser:
    """
    A class for parsing and handling an agent's stream output.
    """

    def __init__(self, callbacks: AgentCallbacks = AgentCallbacks()):
        self.callbacks = callbacks
        self.output = None

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])

    def _process_actions(self, actions: List[Any]) -> None:
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(
                action, "tool"
            ):
                self._process_tool_call(action)

    def _process_tool_call(self, action: Any) -> None:
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observations(self, observations: List[Any]) -> None:
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["observation"] = getattr(
                    observation, "observation", None
                )
            self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        self.callbacks.result_callback(result)
        self.output = result


def pretty_print_messages(messages: list[BaseMessage]):
    for message in messages:
        message.pretty_print()


depth_colors = {
    1: "\033[96m",  # Bright Cyan (First Layer)
    2: "\033[93m",  # Yellow (Second Layer)
    3: "\033[94m",  # Light Green (Third Layer)
    4: "\033[95m",  # Purple (Fourth Layer)
    5: "\033[92m",  # Light Blue (Fifth Layer)
    "default": "\033[96m",  # Default Color
    "reset": "\033[0m",  # Reset Color
}


def is_terminal_dict(data):
    """Check if dictionary is a terminal dictionary."""
    if not isinstance(data, dict):
        return False
    for value in data.values():
        if isinstance(value, (dict, list)) or hasattr(value, "__dict__"):
            return False
    return True


def format_terminal_dict(data):
    """Format terminal dictionary."""
    items = []
    for key, value in data.items():
        if isinstance(value, str):
            items.append(f'"{key}": "{value}"')
        else:
            items.append(f'"{key}": {value}')
    return "{" + ", ".join(items) + "}"


def _display_message_tree(data, indent=0, node=None, is_root=False):
    """Print JSON object tree structure without type information."""
    spacing = " " * indent * 4
    color = depth_colors.get(indent + 1, depth_colors["default"])

    if isinstance(data, dict):
        if not is_root and node is not None:
            if is_terminal_dict(data):
                print(
                    f'{spacing}{color}{node}{depth_colors["reset"]}: {format_terminal_dict(data)}'
                )
            else:
                print(f'{spacing}{color}{node}{depth_colors["reset"]}:')
                for key, value in data.items():
                    _display_message_tree(value, indent + 1, key)
        else:
            for key, value in data.items():
                _display_message_tree(value, indent + 1, key)

    elif isinstance(data, list):
        if not is_root and node is not None:
            print(f'{spacing}{color}{node}{depth_colors["reset"]}:')

        for index, item in enumerate(data):
            print(f'{spacing}    {color}index [{index}]{depth_colors["reset"]}')
            _display_message_tree(item, indent + 1)

    elif hasattr(data, "__dict__") and not is_root:
        if node is not None:
            print(f'{spacing}{color}{node}{depth_colors["reset"]}:')
        _display_message_tree(data.__dict__, indent)

    else:
        if node is not None:
            if isinstance(data, str):
                value_str = f'"{data}"'
            else:
                value_str = str(data)

            print(f'{spacing}{color}{node}{depth_colors["reset"]}: {value_str}')


def display_message_tree(message):
    """Message tree display main function."""
    if isinstance(message, BaseMessage):
        _display_message_tree(message.__dict__, is_root=True)
    else:
        _display_message_tree(message, is_root=True)


class ToolChunkHandler:
    """Class for handling and managing Tool Message chunks."""

    def __init__(self):
        self._reset_state()

    def _reset_state(self) -> None:
        self.gathered = None
        self.first = True
        self.current_node = None
        self.current_namespace = None

    def _should_reset(self, node: str | None, namespace: str | None) -> bool:
        """Check state reset condition"""
        # Default behavior is to not reset if both parameters are None
        if node is None and namespace is None:
            return False

        # If only node is set
        if node is not None and namespace is None:
            return self.current_node != node

        # If only namespace is set
        if namespace is not None and node is None:
            return self.current_namespace != namespace

        # If both are set
        return self.current_node != node or self.current_namespace != namespace

    def process_message(
        self,
        chunk: AIMessageChunk,
        node: str | None = None,
        namespace: str | None = None,
    ) -> None:
        """Process message chunk"""
        if self._should_reset(node, namespace):
            self._reset_state()

        self.current_node = node if node is not None else self.current_node
        self.current_namespace = (
            namespace if namespace is not None else self.current_namespace
        )

        self._accumulate_chunk(chunk)
        return self._display_tool_calls()

    def _accumulate_chunk(self, chunk: AIMessageChunk) -> None:
        """Accumulate chunk content"""
        self.gathered = chunk if self.first else self.gathered + chunk
        self.first = False

    def _display_tool_calls(self) -> None:
        """Display tool calls"""
        if (
            self.gathered
            and not self.gathered.content
            and self.gathered.tool_call_chunks
            and self.gathered.tool_calls
        ):
            return self.gathered.tool_calls[0]["args"]


def get_role_from_messages(msg):
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    else:
        return "assistant"


def messages_to_history(messages):
    return "\n".join(
        [f"{get_role_from_messages(msg)}: {msg.content}" for msg in messages]
    )


def stream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
):
    """
    Stream the output of the LangGraph execution by printing it.

    param: graph (CompiledStateGraph): The compiled LangGraph object to execute
    param: inputs (dict): The input dictionary to pass to the graph
    param: config (RunnableConfig): The execution configuration
    param: node_names (List[str], optional): The list of node names to output. Default is an empty list
    param: callback (Callable, optional): The callback function for each chunk processing. Default is None
    return: None
    """
    prev_node = ""
    for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
        curr_node = metadata["langgraph_node"]

        # Process only if node_names is empty or the current node is in node_names
        if not node_names or curr_node in node_names:
            if callback:
                callback({"node": curr_node, "content": chunk_msg.content})
            else:
                # Print the node name only if it has changed
                if curr_node != prev_node:
                    print("\n" + "=" * 50)
                    print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                    print("- " * 25)
                print(chunk_msg.content, end="", flush=True)

            prev_node = curr_node


def invoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
):
    """
    Pretty-prints the output of the LangGraph execution by streaming it.

    param: graph (CompiledStateGraph): The compiled LangGraph object to execute
    param: inputs (dict): The input dictionary to pass to the graph
    param: config (RunnableConfig): The execution configuration
    param: node_names (List[str], optional): The list of node names to output. Default is an empty list
    param: callback (Callable, optional): The callback function for each chunk processing. Default is None
    return: None
    """

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    for namespace, chunk in graph.stream(
        inputs, config, stream_mode="updates", subgraphs=True
    ):
        for node_name, node_chunk in chunk.items():
            # Filter by node_names if it is not empty
            if len(node_names) > 0 and node_name not in node_names:
                continue

            if callback is not None:
                callback({"node": node_name, "content": node_chunk})
            else:
                print("\n" + "=" * 50)
                formatted_namespace = format_namespace(namespace)
                if formatted_namespace == "root graph":
                    print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                else:
                    print(
                        f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                    )
                print("- " * 25)

                # Print the chunk data of the node
                if isinstance(node_chunk, dict):
                    for k, v in node_chunk.items():
                        if isinstance(v, BaseMessage):
                            v.pretty_print()
                        elif isinstance(v, list):
                            for list_item in v:
                                if isinstance(list_item, BaseMessage):
                                    list_item.pretty_print()
                                else:
                                    print(list_item)
                        elif isinstance(v, dict):
                            for node_chunk_key, node_chunk_value in node_chunk.items():
                                print(f"{node_chunk_key}:\n{node_chunk_value}")
                        else:
                            print(f"\033[1;32m{k}\033[0m:\n{v}")
                else:
                    if node_chunk is not None:
                        for item in node_chunk:
                            print(item)
                print("=" * 50)
