from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv() # Call load_dotenv() at the very beginning

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost") # Default to localhost if not set
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Only if using Qdrant Cloud
COLLECTION_NAME = "startup_proposals"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- 1. Data Loading and Chunking ---
def load_and_chunk_documents(directory="data"): # Changed default directory to 'data'
    documents = []
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist. Please create it and place your PDFs inside.")
        return []

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print(f"Loading {filename}...")
            loader = PyPDFLoader(os.path.join(directory, filename))
            docs = loader.load()
            documents.extend(docs)

    print(f"Loaded {len(documents)} pages in total.") #cite: 2222, 2364, 3206

    # Paragraph chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

# --- 2. Embedding Model ---
class BGEEmbeddings:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True).tolist()[0]

# --- 3. Qdrant Vector Database & Indexing ---
def initialize_qdrant_client():
    if QDRANT_API_KEY:
        client = QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
        )
    else:
        client = QdrantClient(host=QDRANT_HOST, port=6333)
    return client

def index_documents_to_qdrant(chunks, embeddings_model):
    client = initialize_qdrant_client()

    # Create collection if it doesn't exist
    # Consider using client.get_collection() to check existence and skip recreation if not needed for production
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=embeddings_model.model.get_sentence_embedding_dimension(),
                                               distance=models.Distance.COSINE)
        )
        print(f"Collection '{COLLECTION_NAME}' recreated/ensured.")
    except Exception as e:
        print(f"Could not recreate collection, assuming it exists or handling error: {e}")
        # If collection already exists and you don't want to recreate, you might want to clear existing points
        # or handle a different upsert strategy. For this demo, recreate is fine.

    # Prepare points for upsertion
    points = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings_model.embed_query(chunk.page_content)
        # Store source and optionally page number if available in metadata
        source_info = chunk.metadata.get('source', 'unknown')
        page_info = chunk.metadata.get('page', 'N/A')
        points.append(
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={"text": chunk.page_content, "source": f"{source_info} (Page: {page_info})"}
            )
        )

    # Upsert points in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=batch
        )
    print(f"Indexed {len(points)} documents to Qdrant.")

# --- 4. RAG Pipeline Class ---
class RAGPipeline:
    def __init__(self, embeddings_model, groq_client, qdrant_client):
        self.embeddings_model = embeddings_model
        self.groq_client = groq_client
        self.qdrant_client = qdrant_client

    def retrieve_context(self, query, top_k=3):
        query_embedding = self.embeddings_model.embed_query(query)
        search_result = self.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=None
        )
        context = [hit.payload["text"] for hit in search_result]
        sources = [hit.payload["source"] for hit in search_result]
        return "\n\n".join(context), sources

    def generate_response(self, query, context, few_shot_examples=None):
        system_message = {
            "role": "system",
            "content": (
                "You are an AI assistant specialized in startup business proposals. "
                "Answer the user's question based *only* on the provided context. "
                "If the answer is not in the context, state that you cannot find the answer in the provided information. "
                "Be concise and directly answer the question."
            )
        }

        user_query_message = {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }

        messages = [system_message]

        if few_shot_examples:
            for example_query, example_answer in few_shot_examples:
                messages.append({"role": "user", "content": f"Question: {example_query}"})
                messages.append({"role": "assistant", "content": example_answer})

        messages.append(user_query_message)

        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=500
        )
        return chat_completion.choices[0].message.content