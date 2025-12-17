"""
Graph RAG Gradio Frontend

Web interface for querying the Graph RAG system.
Supports local, global, and automatic search modes.
"""

import sys
from pathlib import Path

import gradio as gr
import requests
from loguru import logger
from omegaconf import OmegaConf

# Load config
config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
cfg = OmegaConf.load(config_path)

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Backend URL
DEFAULT_BACKEND_URL = f"http://localhost:{cfg.SERVER.backend_port}"

# Custom CSS
CUSTOM_CSS = """
.answer-box {
    background-color: #1e1e1e;
    border-radius: 8px;
    padding: 16px;
    color: #ffffff;
    font-size: 14px;
    line-height: 1.6;
}
.chain-of-thought {
    background-color: #2d2d2d;
    border-left: 4px solid #4a9eff;
    padding: 12px;
    margin: 8px 0;
    font-style: italic;
    color: #b0b0b0;
}
.context-info {
    background-color: #252525;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    color: #888888;
}
.mode-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
}
.mode-local {
    background-color: #2d5a2d;
    color: #90ee90;
}
.mode-global {
    background-color: #2d4a5a;
    color: #87ceeb;
}
"""


def check_backend_health(backend_url: str) -> dict:
    """Check backend health status."""
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"Status code: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to backend"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_backend(
    question: str,
    mode: str,
    temperature: float,
    max_tokens: int,
    backend_url: str,
) -> tuple[str, str, str, str]:
    """
    Query the backend API.

    Returns:
        Tuple of (chain_of_thought, answer, context_info, prompt)
    """
    if not question.strip():
        return "", "Please enter a question.", "", ""

    try:
        response = requests.post(
            f"{backend_url}/query",
            json={
                "question": question,
                "mode": mode.lower(),
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            return "", f"Error: {error_detail}", "", ""

        data = response.json()

        # Format chain of thought
        chain = data.get("chain_of_thought", "")
        if chain:
            chain = f"**Reasoning:**\n{chain}"

        # Format answer
        answer = data.get("answer", "No answer generated")

        # Format context info
        search_mode = data.get("search_mode", "unknown")
        context_summary = data.get("context_summary", {})

        if search_mode == "local":
            matched = context_summary.get("matched_entities", [])
            nodes = context_summary.get("total_nodes", 0)
            edges = context_summary.get("total_edges", 0)
            context_info = (
                f"**Mode:** Local Search | "
                f"**Matched Entities:** {', '.join(matched) if matched else 'None'} | "
                f"**Nodes:** {nodes} | **Edges:** {edges}"
            )
        else:
            communities = context_summary.get("communities_searched", [])
            total_entities = context_summary.get("total_entities", 0)
            context_info = (
                f"**Mode:** Global Search | "
                f"**Communities:** {len(communities)} | "
                f"**Total Entities:** {total_entities}"
            )

        # Format prompt
        prompt = data.get("prompt", "")

        return chain, answer, context_info, prompt

    except requests.exceptions.ConnectionError:
        return "", "Error: Cannot connect to backend. Is it running?", "", ""
    except requests.exceptions.Timeout:
        return "", "Error: Request timed out. Try again.", "", ""
    except Exception as e:
        logger.error(f"Query error: {e}")
        return "", f"Error: {str(e)}", "", ""


def get_graph_stats(backend_url: str) -> str:
    """Get graph statistics from backend."""
    try:
        response = requests.get(f"{backend_url}/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return (
                f"**Nodes:** {data['num_nodes']} | "
                f"**Edges:** {data['num_edges']} | "
                f"**Communities:** {data['num_communities']}"
            )
        return "Unable to fetch stats"
    except Exception as e:
        return f"Error: {str(e)}"


def create_interface():
    """Create and return the Gradio interface."""
    with gr.Blocks(title="Graph RAG") as demo:
        # Gradio 6.0+: css and theme must be set as attributes
        demo.css = CUSTOM_CSS
        demo.theme = gr.themes.Soft(primary_hue="blue")
        gr.Markdown(
            """
            # Graph RAG Query Interface

            Query a knowledge graph using natural language. The system automatically
            chooses between **local search** (entity-based) and **global search**
            (community-based) depending on your question.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                # Input section
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about the knowledge graph...",
                    lines=2,
                )

                with gr.Row():
                    mode_dropdown = gr.Dropdown(
                        choices=["Auto", "Local", "Global"],
                        value="Auto",
                        label="Search Mode",
                        info="Auto: System decides | Local: Entity traversal | Global: Community summaries",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        label="Temperature",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=128,
                        label="Max Tokens",
                    )

                submit_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=1):
                # Settings
                backend_url = gr.Textbox(
                    label="Backend URL",
                    value=DEFAULT_BACKEND_URL,
                )
                health_btn = gr.Button("Check Health")
                health_output = gr.Markdown("Click 'Check Health' to verify connection")
                stats_output = gr.Markdown("")

        # Output section
        with gr.Row():
            with gr.Column():
                context_info = gr.Markdown(label="Search Context")
                chain_output = gr.Markdown(label="Chain of Thought")
                answer_output = gr.Markdown(label="Answer")

        with gr.Accordion("View Full Prompt", open=False):
            prompt_output = gr.Textbox(
                label="Prompt sent to LLM",
                lines=10,
                interactive=False,
            )

        # Event handlers
        def on_health_check(url):
            health = check_backend_health(url)
            if health.get("status") == "ok":
                stats = get_graph_stats(url)
                return (
                    f"✅ **Connected** | Model: {health.get('llm_model', 'unknown')}",
                    stats,
                )
            else:
                return (
                    f"❌ **Error:** {health.get('message', 'Unknown error')}",
                    "",
                )

        health_btn.click(
            fn=on_health_check,
            inputs=[backend_url],
            outputs=[health_output, stats_output],
        )

        def on_submit(question, mode, temp, max_tok, url):
            chain, answer, context, prompt = query_backend(
                question, mode, temp, max_tok, url
            )
            return chain, answer, context, prompt

        submit_btn.click(
            fn=on_submit,
            inputs=[
                question_input,
                mode_dropdown,
                temperature_slider,
                max_tokens_slider,
                backend_url,
            ],
            outputs=[chain_output, answer_output, context_info, prompt_output],
        )

        # Also trigger on Enter
        question_input.submit(
            fn=on_submit,
            inputs=[
                question_input,
                mode_dropdown,
                temperature_slider,
                max_tokens_slider,
                backend_url,
            ],
            outputs=[chain_output, answer_output, context_info, prompt_output],
        )

        # Example questions
        gr.Examples(
            examples=[
                ["What is AIAP and who runs it?"],
                ["Give me an overview of AI training programs"],
                ["What are the requirements to join AIAP?"],
                ["How long is the AIAP program?"],
            ],
            inputs=question_input,
        )

    return demo


def main():
    """Run the Gradio frontend."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=cfg.SERVER.frontend_port,
        share=False,
    )


if __name__ == "__main__":
    main()
