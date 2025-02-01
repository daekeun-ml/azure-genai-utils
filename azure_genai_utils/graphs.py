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


def visualize_agents(agents, interactions, mermaid_file=None):
    """
    Visuzlie multi-agent interactions using Mermaid. It is useful for visualizing the interactions between agents.

    Args:
        agents (list): List of agent names.
        interactions (list): List of interactions between agents in the format:
                            [("Agent1", "Agent2", "Interaction Description"), ...].
        mermaid_file (str): Filename for the output Mermaid file. If set to not None, the Mermaid code will be saved to this file.
    """
    # Generate Mermaid code
    mermaid_code = "graph TD\n"
    for agent in agents:
        mermaid_code += f"    {agent}[{agent}]\n"
    for src, dst, desc in interactions:
        mermaid_code += f"    {src} -->|{desc}| {dst}\n"

    # Save the Mermaid code to a temporary file
    if mermaid_file is not None:
        with open(mermaid_file, "w", encoding="utf-8") as file:
            file.write(mermaid_code)
        print(f"Graph saved as {mermaid_file}")

    # # Display the generated image in the notebook
    display(
        Image(
            draw_mermaid_png(
                mermaid_code,
                background_color="white",
                draw_method=MermaidDrawMethod.API,
            )
        )
    )
