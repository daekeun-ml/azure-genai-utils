from typing import List, Tuple, Optional
from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_core.runnables.graph_mermaid import draw_mermaid_png


def visualize_langgraph(graph, xray=False, output_file_path=None):
    """
    Visualize LangGraph using Mermaid.
    """
    try:
        if isinstance(graph, CompiledStateGraph):
            display(
                Image(
                    graph.get_graph(xray=xray).draw_mermaid_png(
                        background_color="white",
                        curve_style=CurveStyle.STEP,
                        output_file_path=output_file_path,
                    )
                )
            )
    except (TypeError, ValueError, AttributeError) as e:
        print(f"[ERROR] Visualize LangGraph Error: {e}")


def visualize_agents(
    agents: List[str],
    interactions: List[Tuple[str, str, Optional[str]]],
    mermaid_file: Optional[str] = None,
):
    """
    Visualize multi-agent interactions using Mermaid. It is useful for visualizing the interactions between agents.

    Args:
        agents (List[str]): List of agent names.
        interactions (List[Tuple[str, str, Optional[str]]]): List of interactions between agents in the format:
                            [("Agent1", "Agent2", "Interaction Description"), ...].
        mermaid_file (Optional[str]): Filename for the output Mermaid file. If set to not None, the Mermaid code will be saved to this file.
    """
    # Extracting all agents from interactions
    interaction_agents = set()
    for interaction in interactions:
        if len(interaction) == 2:
            src, dst = interaction
            desc = None
        elif len(interaction) == 3:
            src, dst, desc = interaction
        else:
            raise ValueError(
                "Each interaction must have either 2 or 3 elements (source, destination, optional description)"
            )
        interaction_agents.add(src)
        interaction_agents.add(dst)

    # Check if all agents are used in interactions
    unused_agents = set(agents) - interaction_agents
    if unused_agents:
        raise ValueError(
            f"The following agents are defined but not used in interactions: {', '.join(unused_agents)}"
        )

    # Generate Mermaid code
    mermaid_code = "graph TD\n"

    # Define agents
    for agent in agents:
        mermaid_code += f"    {agent}[{agent}]\n"

    # Define interactions
    for interaction in interactions:
        if len(interaction) == 2:
            src, dst = interaction
            mermaid_code += f"    {src} --> {dst}\n"
        else:
            src, dst, desc = interaction
            mermaid_code += f"    {src} -->|{desc}| {dst}\n"

    # Save the Mermaid code to a file if needed
    if mermaid_file is not None:
        with open(mermaid_file, "w", encoding="utf-8") as file:
            file.write(mermaid_code)
        print(f"Graph saved as {mermaid_file}")

    # Display the generated Mermaid diagram
    display(
        Image(
            draw_mermaid_png(
                mermaid_code,
                background_color="white",
            )
        )
    )
