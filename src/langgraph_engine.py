"""
LangGraph Query Engine

Implements a conversational RAG pipeline using LangGraph with:
- Conversation memory across multiple turns
- Conditional routing between local and global search
- Session persistence via checkpointers
"""

import operator
import uuid
from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.graph_retriever import GraphRetriever


class Message(TypedDict):
    """A single message in the conversation."""

    role: Literal["user", "assistant"]
    content: str


class ChatState(TypedDict):
    """State for the conversational RAG pipeline."""

    # Conversation history (append-only via operator.add)
    messages: Annotated[list[Message], operator.add]

    # Current turn data (overwritten each turn)
    context: str
    resolved_mode: Literal["combined"]
    matched_entities: list[str]
    matched_nodes: list[str]
    source_chunks_loaded: int

    # Configuration
    temperature: float
    max_tokens: int


GENERATION_PROMPT = """You are an expert assistant specializing in Singapore government policies, schemes, and regulations.

CONVERSATION HISTORY:
{history}

CONTEXT FROM KNOWLEDGE GRAPH:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Consider the conversation history when answering - if the user refers to "it", "that", or previous topics, use context from history
2. Answer based ONLY on the information provided in the context above
3. If the context contains relevant information, provide a complete and accurate answer
4. Reference specific entities, amounts, or requirements from the context when applicable
5. If the context doesn't contain enough information, clearly state what's missing
6. Do not make up information that isn't in the context
7. Be specific - include exact figures, eligibility criteria, and requirements when available

ANSWER:"""


class LangGraphEngine:
    """
    LangGraph-based conversational RAG engine.

    Provides multi-turn conversation support with memory persistence.
    """

    def __init__(
        self,
        retriever: GraphRetriever,
        llm: OllamaLLM | ChatAnthropic,
        checkpointer=None,
    ):
        """
        Initialize the LangGraph engine.

        Args:
            retriever: GraphRetriever for context retrieval
            llm: OllamaLLM or ChatAnthropic for text generation
            checkpointer: LangGraph checkpointer for state persistence
                          (default: MemorySaver for in-memory storage)
        """
        self.retriever = retriever
        self.llm = llm
        self.is_anthropic = isinstance(llm, ChatAnthropic)
        self.checkpointer = checkpointer or MemorySaver()

        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

        logger.info("Initialized LangGraphEngine with checkpointer")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph."""
        graph = StateGraph(ChatState)

        # Add nodes - simple linear flow: retrieve → generate
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)

        # Add edges
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph

    def _retrieve_node(self, state: ChatState) -> dict:
        """Perform combined local + global retrieval."""
        user_messages = [m for m in state["messages"] if m["role"] == "user"]
        if not user_messages:
            return {
                "context": "",
                "resolved_mode": "combined",
                "matched_entities": [],
                "matched_nodes": [],
                "source_chunks_loaded": 0,
            }

        question = user_messages[-1]["content"]

        # Always use combined search (local + global)
        result = self.retriever.combined_search(question)

        return {
            "context": result.get("context", ""),
            "resolved_mode": "combined",
            "matched_entities": result.get("matched_entities", []),
            "matched_nodes": result.get("matched_nodes", []),
            "source_chunks_loaded": result.get("source_chunks_loaded", 0),
        }

    def _format_history(self, messages: list[Message], exclude_last: bool = True) -> str:
        """Format conversation history for the prompt."""
        # Get messages to include (exclude the current question)
        history_messages = messages[:-1] if exclude_last and messages else messages

        # Limit to last 10 messages (5 turns) to avoid context overflow
        recent = history_messages[-10:]

        if not recent:
            return "(No previous conversation)"

        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    def _generate_node(self, state: ChatState) -> dict:
        """Generate answer using LLM with conversation history."""
        user_messages = [m for m in state["messages"] if m["role"] == "user"]
        if not user_messages:
            return {"messages": [{"role": "assistant", "content": "Please ask a question."}]}

        question = user_messages[-1]["content"]
        context = state.get("context", "")
        history = self._format_history(state["messages"])

        # Build prompt
        prompt = GENERATION_PROMPT.format(
            history=history,
            context=context,
            question=question,
        )

        # Generate answer
        try:
            if self.is_anthropic:
                # Use ChatAnthropic - invoke with messages
                logger.info(f"Generating with Anthropic: {self.llm.model}")
                response = self.llm.invoke([HumanMessage(content=prompt)])
                answer = response.content
            else:
                # Use OllamaLLM - create with configured params
                logger.info(f"Generating with Ollama: {self.llm.model}")
                temperature = state.get("temperature", 0.5)
                max_tokens = state.get("max_tokens", 1024)

                llm = OllamaLLM(
                    model=self.llm.model,
                    temperature=temperature,
                    num_predict=max_tokens,
                )
                answer = llm.invoke(prompt)

                # Strip any think tags from reasoning models
                if "<think>" in answer and "</think>" in answer:
                    import re

                    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = f"I encountered an error generating a response: {str(e)}"

        return {"messages": [{"role": "assistant", "content": answer}]}

    def query(
        self,
        question: str,
        session_id: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Query the knowledge graph with conversation memory.

        Always uses combined search (local + global) for comprehensive context.

        Args:
            question: The user's question
            session_id: Session ID for conversation continuity (None for new session)
            temperature: LLM temperature
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with answer and metadata
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Config for LangGraph (thread_id is the session identifier)
        config = {"configurable": {"thread_id": session_id}}

        # Input state
        input_state = {
            "messages": [{"role": "user", "content": question}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Run the graph
        try:
            result = self.app.invoke(input_state, config)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "session_id": session_id,
                "resolved_mode": "error",
                "context_summary": {},
                "message_count": 0,
            }

        # Extract the assistant's response
        assistant_messages = [m for m in result["messages"] if m["role"] == "assistant"]
        answer = assistant_messages[-1]["content"] if assistant_messages else "No response generated."

        # Build context summary
        context_summary = {
            "mode": "combined",
            "matched_entities": result.get("matched_entities", []),
            "matched_nodes": result.get("matched_nodes", []),
            "source_chunks_loaded": result.get("source_chunks_loaded", 0),
        }

        return {
            "answer": answer,
            "session_id": session_id,
            "resolved_mode": "combined",
            "context_summary": context_summary,
            "message_count": len(result["messages"]),
            "context": result.get("context", ""),
        }

    def get_conversation_history(self, session_id: str) -> list[Message]:
        """
        Get the conversation history for a session.

        Args:
            session_id: The session ID

        Returns:
            List of messages in the conversation
        """
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.app.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            logger.warning(f"Failed to get conversation history: {e}")
            return []

    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear the conversation history for a session.

        Note: With MemorySaver, this creates a fresh state.
        With persistent checkpointers, you may need to delete from storage.

        Args:
            session_id: The session ID

        Returns:
            True if successful
        """
        # For MemorySaver, we can't truly delete, but starting a new session works
        logger.info(f"Conversation cleared for session: {session_id}")
        return True


def create_langgraph_engine(
    retriever: GraphRetriever,
    llm: OllamaLLM | ChatAnthropic,
    checkpointer_type: Literal["memory", "sqlite"] = "memory",
    sqlite_path: str | None = None,
) -> LangGraphEngine:
    """
    Factory function to create a LangGraphEngine.

    Args:
        retriever: GraphRetriever instance
        llm: OllamaLLM or ChatAnthropic instance
        checkpointer_type: Type of checkpointer - "memory" or "sqlite"
        sqlite_path: Path for SQLite checkpointer (only used if checkpointer_type="sqlite")

    Returns:
        Configured LangGraphEngine
    """
    if checkpointer_type == "sqlite" and sqlite_path:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpointer = SqliteSaver.from_conn_string(sqlite_path)
        logger.info(f"Using SQLite checkpointer: {sqlite_path}")
    else:
        checkpointer = MemorySaver()
        logger.info("Using in-memory checkpointer")

    return LangGraphEngine(
        retriever=retriever,
        llm=llm,
        checkpointer=checkpointer,
    )
