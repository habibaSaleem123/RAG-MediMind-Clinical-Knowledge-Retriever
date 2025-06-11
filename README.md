# 🧠 MedRAG: AI-Powered Clinical Insight Engine

MedRAG is a Retrieval-Augmented Generation (RAG) system designed to answer complex clinical questions by leveraging the MIMIC-IV dataset. It uses dense vector retrieval via FAISS and contextual generation using Google’s Gemini API to provide evidence-backed answers to medical queries.

## 🚀 Features

- ✅ Real-time clinical question answering
- 🔍 Dense semantic retrieval using `SentenceTransformers` and `FAISS`
- 🧬 Overlapping chunking for better context representation
- 💡 Uses `gemini-1.5-flash` for natural language generation
- 📊 Precision and recall evaluation for query relevance
- 🌐 Gradio UI for easy user interaction

## 🏗️ Architecture Overview

1. **Data Sources**: MIMIC-IV diagnosis flowcharts and finished clinical samples
2. **Preprocessing**:
   - Load JSON and TXT medical documents
   - Apply overlapping text chunking (200 words with 50-word overlap)
3. **Embedding & Indexing**:
   - Use `all-mpnet-base-v2` SentenceTransformer for embeddings
   - Store in FAISS index for fast nearest-neighbor search
4. **Query Pipeline**:
   - Encode query → retrieve top-K chunks → construct prompt → generate answer
5. **UI**:
   - Interactive Gradio interface
   - Adjustable top-K slider for chunk retrieval

## 💻 Demo

Launch locally:

```bash
python app.py
# RAG-MediMind-Clinical-Knowledge-Retriever
