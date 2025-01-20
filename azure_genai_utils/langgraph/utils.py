from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.graph import CurveStyle


def visualize_graph(graph, xray=False, output_file_path=None):
    """
    Visualize a state graph.
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
        print(f"[ERROR] Visualize Graph Error: {e}")
