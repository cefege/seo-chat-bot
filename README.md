# SEO Knowledge Assistant

> A production-grade RAG chatbot that answers SEO questions using retrieval-augmented generation — grounding every answer in expert source material with inline citations.

[![Live Demo](https://img.shields.io/badge/Live-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://seo-chat-bot.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-000?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## Table of Contents

- [Why This Project](#why-this-project)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
- [Prompt Engineering](#prompt-engineering)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Technical Decisions](#technical-decisions)
- [License](#license)

---

## Why This Project

Most LLM chatbots hallucinate freely. In a domain like SEO — where outdated or wrong advice can tank a site's rankings — that's unacceptable.

This project solves that by building a **Retrieval-Augmented Generation (RAG) pipeline** that:
- Retrieves relevant content from a curated knowledge base of expert SEO articles before generating an answer
- Forces the model to cite its sources with clickable links, so users can verify every claim
- Filters results by relevance score, so low-quality matches never pollute the context
- Maintains conversation memory across turns, so follow-up questions work naturally

The knowledge base is built from content by **Koray Tugberk Gubur** (one of the foremost experts in semantic SEO) and other authoritative sources.

---

## Architecture

```
                          +------------------+
                          |   User Question  |
                          +--------+---------+
                                   |
                          +--------v---------+
                          | OpenAI Embeddings|
                          | (ada-002, 1536d) |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  Pinecone Vector  |
                          |  Similarity Search |
                          |  (top-k=6, filter)|
                          +--------+---------+
                                   |
                     +-------------v--------------+
                     | Relevance Threshold Filter  |
                     | (score >= 0.25 only)        |
                     +-------------+--------------+
                                   |
                     +-------------v--------------+
                     |   Numbered Source Formatting |
                     |   [Source 1] "Title" (URL)  |
                     +-------------+--------------+
                                   |
              +--------------------v---------------------+
              |         Message Assembly                  |
              |  System Prompt + Chat History (4k budget) |
              |  + Augmented Query with Sources           |
              +--------------------+---------------------+
                                   |
                          +--------v---------+
                          |  GPT-4o-mini     |
                          |  (streaming)     |
                          +--------+---------+
                                   |
                     +-------------v--------------+
                     |   Streamed Response with   |
                     |   Inline [[N](URL)] Cites  |
                     +-------------+--------------+
                                   |
                     +-------------v--------------+
                     |   Source Expander Widget    |
                     |   + Thumbs Up/Down Feedback|
                     +----------------------------+
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | OpenAI GPT-4o-mini | Response generation with streaming |
| **Embeddings** | OpenAI text-embedding-ada-002 | 1536-dimensional vector embeddings for semantic search |
| **Vector Database** | Pinecone (Serverless) | Similarity search over chunked documents using dot product |
| **Text Chunking** | LangChain RecursiveCharacterTextSplitter | 500-token chunks with 20-token overlap, tiktoken-based |
| **Tokenization** | tiktoken | Token counting for conversation memory budget management |
| **Frontend** | Streamlit 1.50 | Chat UI with streaming, status indicators, feedback widgets |
| **Web Scraping** | advertools + Scrapy | Sitemap parsing and content extraction for the knowledge base |
| **Deployment** | Streamlit Cloud | Production hosting with secrets management |

---

## Key Features

### Retrieval-Augmented Generation
Every answer is grounded in retrieved source material — not the model's training data. This dramatically reduces hallucination and ensures answers reflect the actual expert content in the knowledge base.

### Inline Source Citations
Responses use `[[N](URL)]` notation, linking directly to the original article. Users can click through to verify any claim. Sources are also displayed in a collapsible expander with relevance percentages.

### Conversation Memory
A sliding-window token budget (4,000 tokens) maintains recent conversation history. Follow-up questions like *"Can you explain that in more detail?"* or *"How does that relate to what you said about E-E-A-T?"* work naturally because the model sees prior turns.

### Relevance Filtering
Not all vector search results are useful. A configurable relevance threshold (default: 0.25) filters out low-scoring matches before they reach the prompt, keeping the context clean and focused.

### Real-Time Pipeline Visibility
Users see a live status indicator during retrieval: *"Searching knowledge base..."* → *"Found 4 relevant sources"* — making the RAG process transparent rather than a black box.

### Response Feedback
Thumbs up/down feedback widgets after each response enable quality tracking.

---

## How the RAG Pipeline Works

### 1. Document Ingestion (Offline)

The `scraper-embedder.py` script handles the one-time knowledge base creation:

```
Sitemaps + URLs → Scrapy Crawl → Extract body_text
    → Chunk (500 tokens, 20 overlap) → Embed (ada-002)
        → Upsert to Pinecone with metadata (title, url, text)
```

- **Sources**: 6 sitemaps + 17 hand-picked expert articles
- **Chunking strategy**: `RecursiveCharacterTextSplitter` with intelligent separators (`\n\n` → `\n` → ` ` → `""`) preserves semantic boundaries
- **Metadata stored per vector**: `title`, `url`, `text`, `chunk` (index)

### 2. Query-Time Retrieval (Live)

```python
query → embed(query) → pinecone.query(top_k=6) → filter(score >= 0.25) → format
```

Each surviving result is formatted as a numbered source block:
```
[Source 1] "Topical Authority in Semantic SEO" (https://oncrawl.com/...)
The actual chunk text with expert content...
```

### 3. Prompt Assembly

The messages array sent to GPT-4o-mini:
```
[system]  Expert SEO assistant persona + citation rules
[user]    Prior question (from history)
[asst]    Prior answer (from history)
...       (sliding window, newest first, up to 4k tokens)
[user]    ### Sources\n{formatted_sources}\n---\n### Question\n{query}
```

### 4. Streaming Response

The response streams token-by-token via `st.write_stream()`, with the model citing sources inline as `[[1](url)]`.

---

## Prompt Engineering

The system prompt is carefully designed for grounded, accessible answers:

```
You are an SEO expert assistant powered by a curated knowledge base...

Response rules:
1. Base every claim on provided sources. Cite inline using [[N](URL)].
2. If sources don't contain the answer, say so honestly.
3. Never fabricate information or citations.
4. Explain jargon in parentheses on first use.
5. Structure longer answers with headers and bullet points.
6. Build on prior conversation context naturally.
```

**Design choices:**
- **"ONLY the source material"** — hard constraint that prevents hallucination
- **Explicit citation format with examples** — GPT-4o-mini follows formatting examples reliably
- **"Define jargon in parentheses"** — makes technical SEO content accessible to beginners
- **Conversation awareness** — enables natural multi-turn dialogue without repetition

---

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key
- Pinecone API key + index

### Installation

```bash
git clone https://github.com/cefege/seo-chat-bot.git
cd seo-chat-bot
pip install -r requirements.txt
```

### Configuration

Create `.streamlit/secrets.toml`:

```toml
[API]
OPEN_AI_API_KEY = "sk-..."
PINECONE_API_KEY = "your-pinecone-key"
PINECONE_INDEX_NAME = "your-index-name"
PINECONE_HOST = "https://your-index-host.pinecone.io"
```

### Run

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` and start asking SEO questions.

---

## Project Structure

```
seo-chat-bot/
├── streamlit_app.py        # Main chatbot application (RAG pipeline + UI)
├── scraper-embedder.py     # Offline: scrape, chunk, embed, upsert to Pinecone
├── case_studies.md          # Sidebar content: curated article & video links
├── requirements.txt         # Production dependencies
└── .streamlit/
    └── secrets.toml         # API keys (gitignored)
```

---

## Technical Decisions

| Decision | Rationale |
|---|---|
| **GPT-4o-mini over GPT-4o** | Same citation quality at ~15x lower cost. The retrieval does the heavy lifting; the LLM just needs to synthesize and cite. |
| **text-embedding-ada-002** | Existing Pinecone index uses these embeddings. Migration to text-embedding-3-small would require full re-indexing. |
| **Dot product metric** | OpenAI embeddings are normalized, so dot product = cosine similarity with slightly faster computation. |
| **500-token chunks** | Small enough for precise retrieval, large enough to preserve context. 20-token overlap prevents information loss at boundaries. |
| **4k token history budget** | Enough for ~5-8 conversation turns. Keeps costs low while enabling meaningful multi-turn dialogue. |
| **Relevance threshold 0.25** | Empirically tuned to filter noise without being too aggressive. Prevents the model from citing irrelevant content. |
| **Streamlit over custom frontend** | Rapid iteration, built-in chat components, one-click cloud deployment. Perfect for a domain-specific tool. |

---

## License

MIT

---

<p align="center">
  Made by <a href="https://www.linkedin.com/in/mihai-mateias/">Mihai Mateias</a> with much love for my brother Koray.
</p>
