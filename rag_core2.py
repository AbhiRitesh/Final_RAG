from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import json

from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv() # Call load_dotenv() at the very beginning

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
        return context, sources

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

    def llm_judge_evaluate(self, query, context, generated_answer):
        """
        Evaluates a generated answer using an LLM-as-a-judge approach.
        The judge rates the answer on faithfulness and helpfulness.
        """
        judge_system_message = {
            "role": "system",
            "content": (
                "You are an impartial judge. Your task is to evaluate a generated answer based on a given question and context. "
                "Rate the answer on two criteria: **faithfulness** and **helpfulness**. "
                "Faithfulness: Does the answer rely *only* on the provided context? Score 1-5, where 5 is perfectly grounded. "
                "Helpfulness: Does the answer directly and effectively address the user's question? Score 1-5, where 5 is a perfect answer. "
                "Provide your response as a JSON object with a 'faithfulness' integer score, a 'helpfulness' integer score, and a 'summary' string explanation. "
                "Example response: {'faithfulness': 4, 'helpfulness': 5, 'summary': 'The answer is mostly grounded and very helpful.'}"
            )
        }

        judge_user_message = {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Context: {context}\n\n"
                f"Generated Answer: {generated_answer}\n\n"
            )
        }
        
        messages = [judge_system_message, judge_user_message]

        # Use a more powerful model for the judge if available, or just a separate call
        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192", # Using a larger, more capable model as the judge
            temperature=0.1, # Keep the temperature low for consistent, deterministic scoring
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        raw_response = chat_completion.choices[0].message.content
        try:
            parsed_response = json.loads(raw_response)
            faithfulness_score = parsed_response.get('faithfulness')
            helpfulness_score = parsed_response.get('helpfulness')
            summary = parsed_response.get('summary')
            
            if not all([isinstance(faithfulness_score, int), isinstance(helpfulness_score, int), isinstance(summary, str)]):
                 raise ValueError("LLM judge did not return the expected JSON format.")

            scores = {'faithfulness': faithfulness_score, 'helpfulness': helpfulness_score}
            return scores, summary

        except (json.JSONDecodeError, ValueError) as e:
            print(f"LLM judge output was not valid JSON: {raw_response}")
            return {'faithfulness': 1, 'helpfulness': 1}, f"Error parsing judge's response: {e}. Raw output: {raw_response}"