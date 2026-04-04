# SEO Q&A Chatbot

> RAG-powered chatbot for Semantic SEO — retrieves knowledge from expert sources and generates grounded answers using GPT-3.5 + Pinecone.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green)

## Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers Semantic SEO questions by searching through a curated knowledge base of expert content. Instead of relying on the model's training data, every answer is grounded in retrieved source material — reducing hallucination and ensuring accuracy.

**Knowledge sources:**
- seobythesea.com
- holisticseo.digital
- Guest posts by Koray Tugberk Gubur

## Tech Stack

`Python` · `OpenAI GPT-3.5` · `Pinecone` · `Streamlit` · `LangChain`

## Architecture

```
User Question
     |
     v
[Embedding] --> [Pinecone Vector Search] --> Top-K Relevant Chunks
                                                    |
                                                    v
                                        [Augmented Prompt + GPT-3.5]
                                                    |
                                                    v
                                            Grounded Answer
```

1. **Document ingestion** — expert SEO content is chunked and embedded into Pinecone vector database
2. **Retrieval** — user questions are embedded and matched against the knowledge base via cosine similarity
3. **Augmented generation** — retrieved chunks are injected into the prompt as context for GPT-3.5
4. **Response** — model generates an answer grounded in the retrieved sources

## Features

- **RAG pipeline** — retrieval-augmented generation with Pinecone vector search
- **Domain-specific knowledge base** — curated expert SEO content, not generic web data
- **Conversational UI** — Streamlit chat interface with message history
- **Grounded answers** — responses are based on retrieved context, not model hallucination

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Set your API keys as environment variables:

```bash
export PINECONE_API_KEY="your-pinecone-key"
export OPEN_AI_API_KEY="your-openai-key"
```

Run the app:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` and start asking SEO questions.

## License

MIT
