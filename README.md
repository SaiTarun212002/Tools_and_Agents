# Tools_and_Agents

An intelligent, AI-powered research assistant that enables users to interactively explore and question research papers (like *"Attention is All You Need"*) using a combination of local LLMs, semantic search, and Wikipedia-based knowledge enhancement.

Built using:
- 🧠 LangChain for agent orchestration
- 📄 ChromaDB for document retrieval
- 💬 Ollama for local LLM inference (`gemma3`)
- 🌐 Wikipedia API for external factual grounding
- 📊 Streamlit for a simple web UI

---

## 🚀 Key Features

- 📄 Ingests and semantically indexes academic papers (PDFs)
- 🔍 Uses vector search (Chroma + Sentence Transformers) for document retrieval
- 🌐 Incorporates Wikipedia API to enrich factual answers
- 🧠 Combines multiple tools using LangChain's agent framework
- 💻 Accessible via a Streamlit web interface.
