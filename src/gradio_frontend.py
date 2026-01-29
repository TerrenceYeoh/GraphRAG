"""
Graph RAG Gradio Frontend

Web interface for querying the Graph RAG system with conversation memory.
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
.session-info {
    font-size: 11px;
    color: #666;
    padding: 4px 8px;
    background-color: #f0f0f0;
    border-radius: 4px;
    margin-bottom: 8px;
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


def query_backend(
    message: str,
    history: list,
    session_id: str | None,
    temperature: float,
    max_tokens: int,
    backend_url: str,
) -> tuple[list, str, str, str]:
    """
    Query the backend API with conversation memory.

    Returns:
        Tuple of (updated_history, session_id, context_info, chain_of_thought)
    """
    if not message.strip():
        return history, session_id or "", "", ""

    try:
        response = requests.post(
            f"{backend_url}/query",
            json={
                "question": message,
                "session_id": session_id if session_id else None,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Error: {error_detail}"})
            return history, session_id or "", "", ""

        data = response.json()

        # Get session ID (may be new if not provided)
        new_session_id = data.get("session_id", session_id or "")

        # Get answer
        answer = data.get("answer", "No answer generated")

        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        # Format context info
        context_summary = data.get("context_summary", {})
        message_count = data.get("message_count", 0)
        matched = context_summary.get("matched_entities", [])
        source_chunks = context_summary.get("source_chunks_loaded", 0)

        context_info = (
            f"**Mode:** Combined (Local + Global) | "
            f"**Matched:** {', '.join(matched[:3]) if matched else 'None'} | "
            f"**Source Chunks:** {source_chunks} | "
            f"**Messages:** {message_count}"
        )

        # Get chain of thought
        chain = data.get("chain_of_thought", "")
        if chain:
            chain = f"**Reasoning:**\n{chain}"

        return history, new_session_id, context_info, chain

    except requests.exceptions.ConnectionError:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Error: Cannot connect to backend. Is it running?"})
        return history, session_id or "", "", ""
    except requests.exceptions.Timeout:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Error: Request timed out. Try again."})
        return history, session_id or "", "", ""
    except Exception as e:
        logger.error(f"Query error: {e}")
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, session_id or "", "", ""


def create_interface():
    """Create and return the Gradio interface."""
    with gr.Blocks(title="Graph RAG Chat") as demo:
        # Set CSS (Gradio 6.0+ compatibility)
        demo.css = CUSTOM_CSS

        # State for session tracking
        session_state = gr.State(value=None)

        gr.Markdown(
            """
            # Graph RAG Chat Interface

            A conversational interface for querying the knowledge graph with **memory**.
            The system remembers your previous questions in the same session.

            Uses **combined search** (local entity traversal + global community summaries) for comprehensive answers.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about the knowledge graph...",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    new_chat_btn = gr.Button("New Chat", variant="secondary")
                    clear_btn = gr.Button("Clear Display")

                # Context info display
                context_info = gr.Markdown(label="Search Context", value="")
                chain_output = gr.Markdown(label="Chain of Thought", value="")

            with gr.Column(scale=1):
                # Settings panel
                gr.Markdown("### Settings")

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

                gr.Markdown("### Connection")

                backend_url = gr.Textbox(
                    label="Backend URL",
                    value=DEFAULT_BACKEND_URL,
                )

                health_btn = gr.Button("Check Health")
                health_output = gr.Markdown("Click 'Check Health' to verify connection")
                stats_output = gr.Markdown("")

                # Session info
                gr.Markdown("### Session")
                session_display = gr.Textbox(
                    label="Session ID",
                    value="(New session)",
                    interactive=False,
                )

        # Example questions
        gr.Markdown("### Example Questions")
        gr.Examples(
            examples=[
                ["What is CPF?"],
                ["What accounts does it have?"],  # Follow-up (uses "it")
                ["What are the withdrawal rules?"],  # Another follow-up
                ["Give me an overview of Singapore government housing schemes"],
                ["What grants are available for first-time homebuyers?"],
            ],
            inputs=msg_input,
            label="Click an example to try it",
        )

        # Event handlers
        def on_health_check(url):
            health = check_backend_health(url)
            if health.get("status") == "ok":
                stats = get_graph_stats(url)
                return (
                    f"✅ **Connected** | Model: {health.get('llm_model', 'unknown')} | LangGraph Memory: Enabled",
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

        def on_submit(message, history, session_id, temp, max_tok, url):
            history, new_session_id, context, chain = query_backend(
                message, history, session_id, temp, max_tok, url
            )
            # Format session display
            session_display_text = new_session_id[:8] + "..." if new_session_id else "(New session)"
            return history, new_session_id, context, chain, session_display_text, ""

        # Submit on button click
        submit_btn.click(
            fn=on_submit,
            inputs=[
                msg_input,
                chatbot,
                session_state,
                temperature_slider,
                max_tokens_slider,
                backend_url,
            ],
            outputs=[chatbot, session_state, context_info, chain_output, session_display, msg_input],
        )

        # Submit on Enter
        msg_input.submit(
            fn=on_submit,
            inputs=[
                msg_input,
                chatbot,
                session_state,
                temperature_slider,
                max_tokens_slider,
                backend_url,
            ],
            outputs=[chatbot, session_state, context_info, chain_output, session_display, msg_input],
        )

        def on_new_chat():
            """Start a new conversation (clear session)."""
            return [], None, "", "", "(New session)"

        new_chat_btn.click(
            fn=on_new_chat,
            outputs=[chatbot, session_state, context_info, chain_output, session_display],
        )

        def on_clear():
            """Clear the display but keep session."""
            return [], "", ""

        clear_btn.click(
            fn=on_clear,
            outputs=[chatbot, context_info, chain_output],
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
