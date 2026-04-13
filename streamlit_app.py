import tiktoken
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
TEMPERATURE = 0.2
RELEVANCE_THRESHOLD = 0.25
TOP_K = 6
MAX_HISTORY_TOKENS = 4_000

SYSTEM_PROMPT = """
You are an SEO expert assistant powered by a curated knowledge base of authoritative SEO content. Your role is to help users understand Search Engine Optimization concepts clearly and accurately.

## How you work
You answer questions using ONLY the source material provided below each question. Each source is labeled [Source N] with a title and URL.

## Response rules
1. Base every claim on the provided sources. Cite sources inline using the format [[N](URL)] where N is the source number and URL is the source URL.
2. If multiple sources support a point, cite all of them: [[1](url1)] [[3](url3)].
3. If the provided sources do not contain enough information to answer the question, say: "I don't have enough information in my knowledge base to answer that fully." Then share whatever partial information the sources do provide.
4. Never fabricate information or citations.
5. Explain technical SEO concepts in plain language. When you must use jargon (like "canonical tags", "crawl budget", "E-E-A-T"), briefly define it in parentheses on first use.
6. Structure longer answers with headers and bullet points for readability.
7. Keep answers focused and practical. After explaining a concept, suggest a concrete next step or action when relevant.
8. When the conversation has prior context, build on it naturally. Reference earlier points when relevant rather than repeating information.
""".strip()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
pinecone_client = Pinecone(api_key=st.secrets["API"]["PINECONE_API_KEY"])
PINECONE_INDEX_NAME = st.secrets["API"]["PINECONE_INDEX_NAME"]
PINECONE_HOST = st.secrets["API"]["PINECONE_HOST"]

client = OpenAI(
    api_key=st.secrets["API"]["OPEN_AI_API_KEY"],
    max_retries=3,
)

_encoder = tiktoken.encoding_for_model(MODEL)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def get_query_embedding(query: str) -> list[float]:
    return (
        client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        .data[0]
        .embedding
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def get_relevant_sources(query_embedding: list[float], index_name: str) -> list[dict]:
    """Query Pinecone and return filtered, numbered source dicts."""
    index = pinecone_client.Index(name=index_name, host=PINECONE_HOST)
    res = index.query(vector=query_embedding, top_k=TOP_K, include_metadata=True)

    sources = []
    for i, item in enumerate(res["matches"], start=1):
        score = item.get("score", 0)
        if score < RELEVANCE_THRESHOLD:
            continue
        meta = item["metadata"]
        sources.append(
            {
                "index": i,
                "title": meta.get("title", "Untitled"),
                "url": meta.get("url", ""),
                "text": meta.get("text", ""),
                "score": score,
            }
        )
    # Re-number after filtering
    for i, s in enumerate(sources, start=1):
        s["index"] = i
    return sources


def format_sources_for_prompt(sources: list[dict]) -> str:
    """Format sources as numbered blocks for the LLM prompt."""
    if not sources:
        return "(No relevant sources found in the knowledge base.)"
    blocks = []
    for s in sources:
        blocks.append(
            f'[Source {s["index"]}] "{s["title"]}" ({s["url"]})\n{s["text"]}'
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
def build_messages(
    history: list[dict], augmented_query: str
) -> list[dict]:
    """Assemble the messages array with system prompt, recent history, and current query."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add recent history within token budget (newest turns first, then reverse)
    history_messages = []
    token_count = 0
    for msg in reversed(history):
        msg_tokens = count_tokens(msg["content"])
        if token_count + msg_tokens > MAX_HISTORY_TOKENS:
            break
        history_messages.append({"role": msg["role"], "content": msg["content"]})
        token_count += msg_tokens
    history_messages.reverse()
    messages.extend(history_messages)

    # Current query with retrieved context
    messages.append({"role": "user", "content": augmented_query})
    return messages


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def display_sources(sources: list[dict]):
    """Render sources in an expander after an assistant message."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} references)"):
        for s in sources:
            st.markdown(
                f'**[{s["index"]}]** [{s["title"]}]({s["url"]}) — relevance: {s["score"]:.0%}'
            )


def display_chat_history():
    """Render all messages from session state, including source expanders."""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(
            message["role"],
            avatar="🤖" if message["role"] == "assistant" else None,
        ):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                display_sources(message["sources"])
                st.feedback("thumbs", key=f"fb_{i}")


def hide_streamlit_chrome():
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] {visibility: hidden;}
        [data-testid="stMainMenu"] {visibility: hidden;}
        .block-container {padding-top: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def read_markdown_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="SEO Knowledge Assistant", page_icon="🔍")
    hide_streamlit_chrome()

    st.title("SEO Knowledge Assistant")
    st.caption(
        "Ask questions about SEO — answers are sourced from expert articles and "
        "case studies by Koray Tugberk Gubur and other authorities."
    )

    # Session state
    st.session_state.setdefault("messages", [])

    # Sidebar
    with st.sidebar:
        st.markdown(read_markdown_file("case_studies.md"))
        st.divider()
        st.markdown(
            "Built by [Mihai Mateias](https://www.linkedin.com/in/mihai-mateias/)",
        )

    # Display chat history
    display_chat_history()

    # Chat input
    query = st.chat_input("Ask any question about SEO...")
    if not query:
        return

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # --- RAG pipeline ---
    try:
        with st.status("Searching knowledge base...", expanded=False) as status:
            query_embedding = get_query_embedding(query)
            status.update(label="Finding relevant sources...")
            sources = get_relevant_sources(query_embedding, PINECONE_INDEX_NAME)
            status.update(
                label=f"Found {len(sources)} relevant source{'s' if len(sources) != 1 else ''}",
                state="complete",
            )
    except Exception as e:
        st.error(f"Something went wrong while searching: {e}")
        return

    # Build prompt
    context_block = format_sources_for_prompt(sources)
    augmented_query = (
        f"### Sources\n{context_block}\n\n---\n\n### Question\n{query}"
    )

    # History for conversation memory (all messages except the one we just appended)
    history = st.session_state.messages[:-1]
    messages = build_messages(history, augmented_query)

    # Stream response
    try:
        with st.chat_message("assistant", avatar="🤖"):
            stream = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=messages,
                stream=True,
            )
            response = st.write_stream(
                chunk.choices[0].delta.content or ""
                for chunk in stream
                if chunk.choices[0].delta.content is not None
            )
            display_sources(sources)
            st.feedback("thumbs", key=f"fb_{len(st.session_state.messages)}")
    except Exception as e:
        st.error(f"Something went wrong while generating the response: {e}")
        return

    # Store assistant message with sources
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )


if __name__ == "__main__":
    main()
