# rag_core.py

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "startup_proposals"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- 1. Data Loading and Chunking ---
def load_and_chunk_documents(directory="data"):
    documents = []
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist. Please create it and place your PDFs inside.")
        return []

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print(f"Loading {filename}...")
            loader = PyPDFLoader(os.path.join(directory, filename))
            docs = loader.load()
            documents.extend(docs)

    print(f"Loaded {len(documents)} pages in total.")

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

    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=embeddings_model.model.get_sentence_embedding_dimension(),
                                               distance=models.Distance.COSINE)
        )
        print(f"Collection '{COLLECTION_NAME}' recreated/ensured.")
    except Exception as e:
        print(f"Could not recreate collection, assuming it exists or handling error: {e}")

    points = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings_model.embed_query(chunk.page_content)
        source_info = chunk.metadata.get('source', 'unknown')
        page_info = chunk.metadata.get('page', 'N/A')
        points.append(
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={"text": chunk.page_content, "source": f"{source_info} (Page: {page_info})"}
            )
        )

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

    def evaluate_with_judge(self, query, answer, context):
        """
        Uses an LLM as a judge to evaluate a generated answer.
        The judge will assess the answer's correctness and faithfulness to the context.
        """
        judge_system_prompt = {
            "role": "system",
            "content": (
                "You are an expert evaluator. Your task is to act as an impartial judge to evaluate a generated answer "
                "based on a given question and a provided context. "
                "You must perform two checks: "
                "1. **Faithfulness**: Does the answer contain information that is directly supported by the context? "
                "2. **Correctness**: Does the answer directly and accurately address the user's question, using only the provided context? "
                "Provide a final verdict (e.g., 'Correct', 'Incorrect', 'Partially Correct') and a brief reasoning for your decision. "
                "The format should be: 'Verdict: [Your Verdict]\nReasoning: [Your Reasoning]'"
            )
        }

        judge_user_prompt = {
            "role": "user",
            "content": f"""
Question: {query}

Context:
{context}

Generated Answer:
{answer}

---
Evaluate the generated answer based on the criteria above.
"""
        }

        messages = [judge_system_prompt, judge_user_prompt]
        
        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192", # Using the same model for the judge
            temperature=0.0, # Use a low temperature for deterministic evaluation
            max_tokens=250
        )

        response_text = chat_completion.choices[0].message.content
        
        # Parse the verdict and reasoning from the LLM's response
        verdict = "Could not parse verdict."
        reasoning = "Could not parse reasoning."

        if "Verdict:" in response_text and "Reasoning:" in response_text:
            lines = response_text.split('\n')
            for line in lines:
                if line.startswith("Verdict:"):
                    verdict = line.replace("Verdict:", "").strip()
                elif line.startswith("Reasoning:"):
                    reasoning = line.replace("Reasoning:", "").strip()
        else:
            reasoning = response_text # If parsing fails, use the entire response as reasoning
        
        return verdict, reasoning