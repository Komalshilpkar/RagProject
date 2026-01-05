import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from transformers import pipeline

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Advanced RAG PDF Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Advanced RAG PDF Chatbot (Industry Level)")
st.caption("Local • Offline • Manual Memory • FAISS • HuggingFace")

# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("⚙️ Controls")
top_k = st.sidebar.slider("Top-K Documents", 1, 5, 3)
chunk_size = st.sidebar.selectbox("Chunk Size", [300, 500, 700], index=1)

# ===============================
# FILE UPLOADER
# ===============================
uploaded_file = st.file_uploader(
    "📄 Upload a PDF",
    type="pdf",
    key="pdf_upload"
)

if uploaded_file is None:
    st.info("👆 Please upload a PDF to start chatting")

else:
    # Save PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # ===============================
    # LOAD DOCUMENT
    # ===============================
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # ===============================
    # ADD METADATA
    # ===============================
    documents = []
    for i, page in enumerate(pages):
        documents.append(
            Document(
                page_content=page.page_content,
                metadata={
                    "page": i + 1,
                    "source": uploaded_file.name
                }
            )
        )

    # ===============================
    # TEXT SPLITTING
    # ===============================
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(documents)

    # ===============================
    # EMBEDDINGS
    # ===============================
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ===============================
    # VECTOR DATABASE
    # ===============================
    st.session_state.vectorstore = FAISS.from_documents(
        split_docs, embeddings
    )

    st.success("✅ Document indexed successfully!")

# ===============================
# LOAD LOCAL LLM
# ===============================
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256
    )
    return HuggingFacePipeline(pipeline=pipe)

# ===============================
# QUESTION ANSWERING
# ===============================
if st.session_state.vectorstore is not None:
    llm = load_llm()

    query = st.text_input("💬 Ask a question")

    if query:
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )

        # ✅ NEW RETRIEVER API
        docs = retriever.invoke(query)

        # ===============================
        # CONTEXT
        # ===============================
        context = ""
        for d in docs:
            context += f"\n(Page {d.metadata['page']}): {d.page_content}"

        # ===============================
        # MEMORY
        # ===============================
        history = "\n".join(st.session_state.chat_history)

        # ===============================
        # PROMPT
        # ===============================
        prompt = f"""
You are an intelligent assistant.
Answer ONLY from the provided context.
If answer is not present, say: "Not found in document".

Conversation History:
{history}

Context:
{context}

Question:
{query}
"""

        # ✅ FIXED HERE
        answer = llm.invoke(prompt)

        # Save conversation
        st.session_state.chat_history.append(f"User: {query}")
        st.session_state.chat_history.append(f"Assistant: {answer}")

        # ===============================
        # OUTPUT
        # ===============================
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("✅ Answer")
            st.write(answer)

        with col2:
            st.subheader("📚 Source Pages")
            for d in docs:
                st.write(f"Page {d.metadata['page']}")


