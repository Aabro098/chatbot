import os
import random
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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
    """Initialize the QA chain for chatbot"""
    # Load FAISS database
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS database: {e}")

    retriever = db.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
        | load_llm(HF_MODEL_ID)
        | StrOutputParser()
    )

    return qa_chain


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
        return qa_chain.invoke(query_text)
