import os
import random
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())


class QueryRequest(BaseModel):
    query: str = None


class QueryResponse(BaseModel):
    query: str
    response: str


# Setup Hugging Face hosted model
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.environ.get("HF_TOKEN")
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", 8))
FINAL_CONTEXT_TOP_K = max(3, min(5, int(os.environ.get("FINAL_CONTEXT_TOP_K", 4))))
RERANK_SNIPPET_CHARS = int(os.environ.get("RERANK_SNIPPET_CHARS", 350))
CONTEXT_SNIPPET_CHARS = int(os.environ.get("CONTEXT_SNIPPET_CHARS", 800))
CONTEXT_COMPRESSION_MAX_CHARS = int(os.environ.get("CONTEXT_COMPRESSION_MAX_CHARS", 1200))
RERANK_MODEL_ID = os.environ.get("RERANK_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful AI assistant specialized in technology skills, career development, and technical topics.

Use the pieces of information provided in the context to answer the user's question.
If relevant context is provided, prioritize using it in your answer.
If no context is available, you may answer based on your general knowledge, BUT ONLY if the question is related to:
- Technology skills (programming, frameworks, databases, cloud, DevOps, etc.)
- Tech career development
- Technical concepts and best practices
- Learning resources and career paths in tech

For ANY question that is NOT related to technology, tech skills, or career development in tech:
- Politely decline and redirect the user back to tech-related topics

Context: {context}
Question: {question}

Guidelines:
- Keep answers concise and direct
- Focus only on tech-related topics
- If uncertain about the answer, be honest and say you don't know
- Do NOT provide information outside the technology and skills domain
- Start the answer directly without small talk or greetings
"""

CONTEXT_COMPRESSION_PROMPT_TEMPLATE = """
You are compressing retrieved context for a retrieval-augmented generation system.

Question: {question}

Context chunks:
{context}

Task:
- Compress the context into a short, question-focused summary.
- Keep only facts that help answer the question.
- Remove repetition, filler, and unrelated details.
- Preserve technical terms and important entities.
- Output at most {max_chars} characters.
- Use compact bullet points if helpful.
- Do not add any commentary or preamble.

Compressed context:
"""

def load_llm(huggingface_repo_id):
    """Load the Hugging Face chat model"""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in the environment")
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.4,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=256,
    )
    return ChatHuggingFace(llm=llm, max_tokens=256)


def set_custom_prompt(custom_prompt_template):
    """Create a chat prompt template"""
    return ChatPromptTemplate.from_template(custom_prompt_template)


def load_chatbot_chain():
    """Initialize the QA pipeline for chatbot"""
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS database: {e}")

    retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    llm = load_llm(HF_MODEL_ID)
    answer_chain = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant specialized in technology skills, career development, and technical topics."),
        ("human", CUSTOM_PROMPT_TEMPLATE),
    ]) | llm | StrOutputParser()
    compression_chain = ChatPromptTemplate.from_messages([
        ("system", "You compress retrieved context for retrieval-augmented generation."),
        ("human", CONTEXT_COMPRESSION_PROMPT_TEMPLATE),
    ]) | llm | StrOutputParser()
    cross_encoder = CrossEncoder(RERANK_MODEL_ID)

    def truncate_text(text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        return normalized[:limit]

    def format_doc_for_rerank(doc, index: int) -> str:
        source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
        prefix = f"Chunk {index}"
        if source:
            prefix += f" | Source: {source}"
        return f"{prefix}\n{truncate_text(doc.page_content, RERANK_SNIPPET_CHARS)}"

    def format_doc_for_context(doc) -> str:
        source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
        prefix = f"Source: {source}\n" if source else ""
        return f"{prefix}{truncate_text(doc.page_content, CONTEXT_SNIPPET_CHARS)}"

    def rerank_documents(question: str, docs):
        if len(docs) <= FINAL_CONTEXT_TOP_K:
            return docs[:FINAL_CONTEXT_TOP_K]

        pairs = [(question, doc.page_content) for doc in docs]
        scores = cross_encoder.predict(pairs)
        ranked_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        selected_docs = [docs[index] for index in ranked_indices[:FINAL_CONTEXT_TOP_K]]

        if not selected_docs:
            return docs[:FINAL_CONTEXT_TOP_K]

        return selected_docs

    def answer_question(question: str) -> str:
        retrieved_docs = retriever.invoke(question)
        selected_docs = rerank_documents(question, retrieved_docs)
        raw_context = "\n\n".join(format_doc_for_context(doc) for doc in selected_docs)
        context = compression_chain.invoke(
            {
                "question": question,
                "context": raw_context,
                "max_chars": CONTEXT_COMPRESSION_MAX_CHARS,
            }
        )
        context = context[:CONTEXT_COMPRESSION_MAX_CHARS]
        return answer_chain.invoke({"context": context, "question": question})

    return answer_question


def is_greeting(text: str) -> bool:
    """Check if the input text is a greeting"""
    greetings_keywords = {
        "hi", "hello", "hey",
        "good morning", "good afternoon", "good evening"
    }
    return text.lower().strip() in greetings_keywords


def get_greeting_response(user_text: str) -> str:
    """Generate a greeting response based on user's greeting"""
    greeting_responses = [
        "Hello! I'm SkillBot. How can I help you today?",
        "Hi there! This is SkillBot. What would you like to know?",
        "Greetings! I'm SkillBot, your helpful assistant. How can I assist you?",
        "Hey! I'm SkillBot. Ready to answer your questions!"
    ]
    return random.choice(greeting_responses)


# Initialize QA chain globally
qa_chain = load_chatbot_chain()


def process_query(query_text: str) -> str:
    """Process a user query and return response"""
    if is_greeting(query_text):
        return get_greeting_response(query_text)
    else:
        return qa_chain(query_text)
