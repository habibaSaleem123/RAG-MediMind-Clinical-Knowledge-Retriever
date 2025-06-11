🧾 README.md
# 🧠 MediMind: Clinical Knowledge Retriever

**MediMind** is an advanced Retrieval-Augmented Generation (RAG) system tailored for clinical question-answering using the MIMIC-IV extended dataset. It combines semantic search with large language models (LLMs) to provide contextual, evidence-backed responses to complex medical queries.

---

## 🚀 Features

- 🔍 **Semantic Chunk Retrieval** with FAISS & Sentence Transformers
- 🧬 **Overlapping Chunking** for richer context windows
- 🤖 **LLM Integration** using Google Gemini 1.5 Flash
- 📊 **Evaluation Metrics**: Precision@K, Recall@K
- 🖥️ **Gradio UI** for interactive clinical queries

---

## 📦 Setup

## 1. Install dependencies
pip install -q sentence-transformers faiss-cpu google-generativeai pandas gradio
## 2. Set up your environment
Ensure your Google API key is configured:
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")

### 3. Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

### 🧩 Data Preprocessing
Load and combine flowcharts (.json) and clinical notes (.txt/.json) from MIMIC-IV extensions.
Chunk documents using overlapping word windows (default 200 words with 50-word overlap).
Generate dense embeddings using all-mpnet-base-v2.

### 📚 RAG Pipeline
def rag_pipeline(query, top_k=10):
    embedding = model.encode([query])[0].astype("float32")
    distances, indices = index.search(np.array([embedding]), top_k)
    chunks = chunked_df.iloc[indices[0]]["chunk"].tolist()
    context = " ".join(chunks)

    prompt = f"""
    You are a clinical assistant AI. Use the context below to answer the user's question clearly.
    ---
    {context}
    ---
    Question: {query}
    """
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    return response.text, chunks
    
### 🧪 Evaluation
Supports:
Precision@K
Recall@K
Sample usage:
precision, recall = evaluate_rag("How is hypertension diagnosed?", ["blood pressure", "hypertension"])
print(f"Precision@10: {precision:.2f}, Recall@10: {recall:.2f}")

### 🌐 Gradio Interface
Launch with:
gr.Interface(
    fn=interface,
    inputs=[
        gr.Textbox(label="Enter Clinical Question"),
        gr.Slider(minimum=5, maximum=20, value=10, label="Top-K Chunks")
    ],
    outputs=[
        gr.Textbox(label="AI Answer", lines=5),
        gr.Textbox(label="Retrieved Chunks", lines=15)
    ],
    title="🧠 MediMind: Clinical Knowledge Retriever",
    description="Ask clinical questions and get AI-generated answers based on the MIMIC-IV dataset."
).launch()


### 📁 Directory Structure
├── mimic_iv_chunks.index        # FAISS index
├── chunked_data.pkl             # Pickled DataFrame of text chunks
├── modelTrain.ipynb             # Notebook for preprocessing & training
├── app.py                       # RAG + Gradio interface
└── README.md

### 🔐 Security
Replace hardcoded API keys with environment variables for production deployment.
### 🧑‍⚕️ Use Cases
Clinical education & training
Decision support for medical professionals
Retrieval QA for MIMIC-style datasets

### 📜 License
This project is for educational and research use only. Use responsibly with clinical data.

### 🤝 Contributing
Pull requests and suggestions are welcome!

### 🌟 Acknowledgements
Built using:
MIMIC-IV Clinical Dataset
Google Gemini
FAISS by Meta
Hugging Face Sentence Transformers

---

Let me know if you'd like to containerize this project with Docker or deploy it on platforms like Hugging Face Spaces or Streamlit Cloud.
