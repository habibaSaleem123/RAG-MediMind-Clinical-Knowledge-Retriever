import gradio as gr
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load preprocessed data and index
chunked_df = pd.read_pickle("chunked_data.pkl")
index = faiss.read_index("mimic_iv_chunks.index")
model = SentenceTransformer("all-mpnet-base-v2")

# Set your Gemini API key
genai.configure(api_key="AIzaSyBZ7omsVAfybOHdTSDZ04a5E5TaaZDuBOk")  # Replace with secure method if deployed

# Define the RAG pipeline function
def rag_query(query, top_k=10):
    embedding = model.encode([query])[0].astype("float32")
    distances, indices = index.search(np.array([embedding]), top_k)
    retrieved_chunks = chunked_df.iloc[indices[0]]["chunk"].tolist()
    context = " ".join(retrieved_chunks)

    prompt = f"""
    You are a clinical assistant AI. Use the context below to provide a clear answer.
    ---
    {context}
    ---
    Question: {query}
    """
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    return response.text, "\n\n".join([f"🔹 {c}" for c in retrieved_chunks])

# Gradio interface
def interface(query, top_k):
    answer, chunks = rag_query(query, top_k)
    return answer, chunks

gr.Interface(
    fn=interface,
    inputs=[
        gr.Textbox(label="Enter Clinical Question", placeholder="e.g., How is hypertension diagnosed and treated?"),
        gr.Slider(minimum=5, maximum=20, step=1, value=10, label="Top-K Chunks to Retrieve")
    ],
    outputs=[
        gr.Textbox(label="AI Answer", lines=5),
        gr.Textbox(label="Retrieved Chunks", lines=15)
    ],
    title="🧠 RAG-based Clinical Assistant",
    description="Ask clinical questions and get AI-generated answers based on the MIMIC-IV diagnostic knowledge."
).launch(debug=True)
