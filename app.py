# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from sklearn.metrics import f1_score
import re
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Import core RAG functionalities
from rag_core import (
    load_and_chunk_documents,
    BGEEmbeddings,
    initialize_qdrant_client,
    index_documents_to_qdrant,
    RAGPipeline,
    COLLECTION_NAME
)

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Startup Business Proposal RAG", layout="wide")
st.title("Startup Business Proposal Q&A 🚀")

# --- Environment Variable Check ---
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY environment variable not set. Please set it to proceed.")
    st.stop()
if not os.getenv("QDRANT_HOST") or not os.getenv("QDRANT_API_KEY"):
    st.error("QDRANT_HOST or QDRANT_API_KEY environment variables not set. Please set them to proceed.")
    st.stop()

# --- Initialize Session State ---
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'few_shot_examples' not in st.session_state:
    st.session_state.few_shot_examples = [
        ("What are the main purposes of a business plan?", "The three main purposes of a business plan are to establish a business focus, secure funding, and attract executives."),
        ("What should be included in the company description?", "The company description should include brief information such as when the company was founded, its business entity type (LLC, C corporation, or S corporation), the state(s) it is registered in, and a summary of its history."),
        ("What is a mission statement?", "A mission statement is a quick explanation of your company's reason for existence, often as short as a tagline, ideally limited to one or two sentences.")
    ]
if 'current_generated_answer' not in st.session_state:
    st.session_state.current_generated_answer = ""
if 'current_expected_answer' not in st.session_state:
    st.session_state.current_expected_answer = ""
if 'current_f1_score' not in st.session_state:
    st.session_state.current_f1_score = None


# --- Helper for F1-score tokenization ---
def tokenize_for_f1(text):
    # Convert to lowercase and split by non-alphanumeric characters, then filter empty strings
    return [word for word in re.split(r'\W+', text.lower()) if word]


# --- Sidebar for Settings and Initialization ---
with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("PDF Documents Directory", value="data")

    if st.button("Initialize RAG System"):
        with st.spinner("Initializing RAG system (this may take a while)..."):
            try:
                # 1. Load and Chunk Documents
                st.write("Loading and chunking documents...")
                all_chunks = load_and_chunk_documents(data_dir)
                if not all_chunks:
                    st.error("No documents loaded. Please check the directory and PDF files.")
                    st.session_state.rag_initialized = False
                    st.stop()

                # 2. Initialize Embedding Model
                st.write("Initializing embedding model...")
                embeddings_model = BGEEmbeddings()
                st.session_state.embeddings_model = embeddings_model

                # 3. Initialize Qdrant Client
                st.write("Initializing Qdrant client...")
                qdrant_client = initialize_qdrant_client()
                st.session_state.qdrant_client = qdrant_client

                # 4. Index Documents to Qdrant
                st.write(f"Indexing documents to Qdrant collection: {COLLECTION_NAME}...")
                index_documents_to_qdrant(all_chunks, embeddings_model)

                # 5. Initialize Groq Client
                st.write("Initializing Groq client...")
                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                st.session_state.groq_client = groq_client

                # 6. Initialize RAG Pipeline
                st.write("Initializing RAG pipeline...")
                rag_pipeline = RAGPipeline(embeddings_model, groq_client, qdrant_client)
                st.session_state.rag_pipeline = rag_pipeline

                st.session_state.rag_initialized = True
                st.success("RAG System Initialized Successfully!")

            except Exception as e:
                st.error(f"Error during RAG system initialization: {e}")
                st.session_state.rag_initialized = False


    st.subheader("Few-Shot Examples (for LLM Guidance)")
    st.write("These examples help guide the LLM's response style.")
    for i, (q, a) in enumerate(st.session_state.few_shot_examples):
        st.text_area(f"Example {i+1} Question:", value=q, key=f"fs_q_{i}_sidebar") # Unique keys
        st.text_area(f"Example {i+1} Answer:", value=a, key=f"fs_a_{i}_sidebar") # Unique keys
        if st.button(f"Remove Example {i+1}", key=f"remove_fs_{i}_sidebar"): # Unique keys
            st.session_state.few_shot_examples.pop(i)
            st.experimental_rerun()
    
    new_q = st.text_input("New Few-Shot Question:", key="new_fs_q_sidebar") # Unique keys
    new_a = st.text_input("New Few-Shot Answer:", key="new_fs_a_sidebar") # Unique keys
    if st.button("Add Few-Shot Example", key="add_fs_button_sidebar"): # Unique key
        if new_q and new_a:
            st.session_state.few_shot_examples.append((new_q, new_a))
            st.experimental_rerun()
        else:
            st.warning("Please enter both question and answer for a new example.")


# --- Main Application Logic ---
if not st.session_state.rag_initialized:
    st.info("Please initialize the RAG system from the sidebar.")
else:
    st.subheader("Ask a Question")
    user_query = st.text_area("Your Question:", height=100, key="user_query_input")
    
    # Use a flag in session_state to control when to display the answer and evaluation
    if st.button("Get Answer", key="get_answer_button"):
        if user_query:
            with st.spinner("Searching for context and generating answer..."):
                try:
                    context, sources = st.session_state.rag_pipeline.retrieve_context(user_query)
                    
                    context_display = "\n\n---\n\n".join(context)
                    source_display = ", ".join(sources) if sources else "No specific sources found."

                    response = st.session_state.rag_pipeline.generate_response(user_query, context, st.session_state.few_shot_examples)

                    # Store the generated answer and reset expected/F1 for new query
                    st.session_state.current_generated_answer = response
                    st.session_state.current_sources = source_display
                    st.session_state.current_context_display = context_display
                    st.session_state.current_expected_answer = "" # Clear previous expected answer
                    st.session_state.current_f1_score = None # Clear previous F1 score

                except Exception as e:
                    st.error(f"Error getting answer: {e}")
                    st.session_state.current_generated_answer = ""
                    st.session_state.current_sources = ""
                    st.session_state.current_context_display = ""
                    st.session_state.current_expected_answer = ""
                    st.session_state.current_f1_score = None
        else:
            st.warning("Please enter a question to get an answer.")

    # Always display the last generated answer and evaluation section if an answer exists
    if st.session_state.current_generated_answer:
        st.success("Answer Generated!")
        st.write("### Answer:")
        st.write(st.session_state.current_generated_answer)
        st.write(f"**Sources:** {st.session_state.current_sources}")

        with st.expander("See Retrieved Context"):
            st.text_area("Context:", value=st.session_state.current_context_display, height=300, disabled=True)

        # --- Evaluation Section ---
        st.write("### Evaluate Answer (Manual F1-Score)")
        # Store the expected_answer input in session_state as it's typed
        st.session_state.current_expected_answer = st.text_area(
            "Enter Expected Answer for F1 Calculation:",
            value=st.session_state.current_expected_answer, # Load from session_state
            key="eval_expected_answer_input" # Unique key for this widget
        )
        
        # Calculate F1-Score button
        if st.button("Calculate F1-Score", key="calculate_f1_button"):
            if st.session_state.current_expected_answer:
                predicted_tokens = tokenize_for_f1(st.session_state.current_generated_answer)
                expected_tokens = tokenize_for_f1(st.session_state.current_expected_answer)

                common_tokens = set(predicted_tokens) & set(expected_tokens)
                
                precision = len(common_tokens) / len(predicted_tokens) if predicted_tokens else 0
                recall = len(common_tokens) / len(expected_tokens) if expected_tokens else 0

                if (precision + recall) == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                
                st.session_state.current_f1_score = f1 # Store F1 in session state
            else:
                st.warning("Please provide an expected answer to calculate F1-Score.")
                st.session_state.current_f1_score = None # Clear F1 if no expected answer

        # Display F1-score if it has been calculated
        if st.session_state.current_f1_score is not None:
            st.metric(label="F1-Score", value=f"{st.session_state.current_f1_score:.4f}")
            st.info(
                "The F1-score is the harmonic mean of precision and recall. "
                "A higher score indicates a better balance between correctly identifying relevant information (precision) "
                "and retrieving all relevant information (recall) in the generated answer compared to the expected answer."
            )