import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 Book RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a book and ask questions from it</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📖 Upload Book")

    uploaded_file = st.file_uploader(
        "Upload your PDF book",
        type=["pdf"]
    )

    st.divider()

    st.subheader("⚙️ RAG Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50
    )

    top_k = st.slider(
        "Retrieved Documents",
        min_value=1,
        max_value=10,
        value=4
    )

    st.divider()

    if st.session_state.book_name:
        st.success(
            f"📚 Current book:\n\n"
            f"{st.session_state.book_name}"
        )


# --------------------------------------------------
# Process Uploaded PDF
# --------------------------------------------------

if uploaded_file is not None:

    if st.session_state.book_name != uploaded_file.name:

        with st.spinner("📚 Processing your book..."):

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.getbuffer())
                pdf_path = tmp_file.name

            try:

                # Load PDF
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                st.info(
                    f"📄 Loaded {len(docs)} pages"
                )

                # Split documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                chunks = splitter.split_documents(docs)

                st.info(
                    f"🔹 Created {len(chunks)} chunks"
                )

                # Embedding model
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Create unique Chroma directory
                db_path = os.path.join(
                    "chroma_db",
                    uploaded_file.name.replace(".pdf", "")
                )

                # Remove old database for this book
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)

                # Create vector database
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory=db_path
                )

                # Save in session
                st.session_state.vectorstore = vectorstore
                st.session_state.book_name = uploaded_file.name
                st.session_state.messages = []

                st.success(
                    "✅ Book processed successfully!"
                )

            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF:\n\n{e}"
                )

            finally:

                # Remove temporary PDF
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)


# --------------------------------------------------
# Chat Interface
# --------------------------------------------------

if st.session_state.vectorstore is None:

    st.info(
        "👈 Please upload a PDF book from the sidebar "
        "to start asking questions."
    )

else:

    st.subheader(
        f"💬 Ask questions about: "
        f"{st.session_state.book_name}"
    )

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    # Chat input
    query = st.chat_input(
        "Ask something about your book..."
    )

    if query:

        # Display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        # --------------------------------------------------
        # Retriever
        # --------------------------------------------------

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        with st.chat_message("assistant"):

            with st.spinner("🤔 Searching the book..."):

                try:

                    # Retrieve relevant chunks
                    docs = retriever.invoke(query)

                    context = "\n\n".join(
                        [
                            doc.page_content
                            for doc in docs
                        ]
                    )

                    # --------------------------------------------------
                    # LLM
                    # --------------------------------------------------

                    llm = ChatMistralAI(
                        model="ministral-3b-2512"
                    )

                    # --------------------------------------------------
                    # Prompt
                    # --------------------------------------------------

                    prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                """
                                You are a helpful AI assistant.

                                Answer the question using ONLY
                                the provided context.

                                If the answer is not present
                                in the context, say:

                                "I could not find the answer
                                in the document."

                                Do not use outside knowledge.
                                """
                            ),
                            (
                                "human",
                                """
                                Context:

                                {context}

                                Question:

                                {question}
                                """
                            )
                        ]
                    )

                    # Create final prompt
                    final_prompt = prompt.invoke(
                        {
                            "context": context,
                            "question": query
                        }
                    )

                    # Generate answer
                    response = llm.invoke(
                        final_prompt
                    )

                    answer = response.content

                    st.markdown(answer)

                    # Save assistant response
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"❌ Error:\n\n{e}"
                    )