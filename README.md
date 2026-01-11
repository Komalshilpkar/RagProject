# RagProject
# Advanced RAG PDF Chatbot (Industry-Level)

An **end-to-end Retrieval-Augmented Generation (RAG) application** that lets users upload a PDF and ask questions.  
The system retrieves **relevant document chunks** using semantic search and generates **accurate, grounded answers** using a **local LLM**.

✅ Offline  
✅ No paid APIs  
✅ Latest LangChain APIs  
✅ Industry-safe architecture  

---

## 🔗 Project Link

- 🌐 **Live App**: https://ragprojec.streamlit.app/

---

## Features

- Upload any PDF document
- Semantic search using FAISS vector database
- Retrieval-Augmented Generation (RAG)
- Conversational memory (manual, stable)
- Source page tracking for answers
- Interactive web UI using Streamlit
- Fully local & offline (no OpenAI / paid APIs)

---

## 🧠 Architecture Overview

PDF → Chunking → Embeddings → FAISS Vector DB
↓
Retriever (Top-K)
↓
Context + Chat History
↓
Local LLM (FLAN-T5)
↓
Final Answer



---

## 🛠️ Tech Stack

- Python
- Streamlit – UI
- LangChain (latest) – RAG framework
- FAISS – Vector database
- HuggingFace Transformers
- Sentence-Transformers – Embeddings
- FLAN-T5 (small) – Local LLM

---

## 📁 Project Structure

RagProject/
│── app.py
│── README.md
│── requirements.txt

👩‍💻 Author

Nandini Shilpkar
