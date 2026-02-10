import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
import uvicorn

# Load environment variables
load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI(
    title="SkillBot API",
    description="API for querying documents using RAG with LLM",
    version="1.0.0"
)

# Request and Response models
class QueryRequest(BaseModel):
    query: str = None
    
class QueryResponse(BaseModel):
    query: str
    response: str

# Setup LLM
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.4,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=1024
    )
    chat_llm = ChatHuggingFace(llm=llm)
    return chat_llm

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    return prompt

# Load FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS database: {e}")

# Create QA chain
retriever = db.as_retriever(search_kwargs={'k': 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    | load_llm(HUGGINGFACE_REPO_ID)
    | StrOutputParser()
)

# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - Health check"""
    return {"message": "Chatbot API is running", "status": "healthy"}

def is_greeting(text: str) -> bool:
    """Check if the text contains greeting keywords"""
    greetings_keywords = [
        "hi", "hello", "hey", "greetings", "good morning", 
        "good afternoon", "good evening", "what's up", "whats up",
        "howdy", "hola", "namaste", "welcome"
    ]
    text_lower = text.lower().strip()
    return any(greeting in text_lower for greeting in greetings_keywords)

def get_greeting_response(user_text: str) -> str:
    """Generate a greeting response based on user's greeting"""
    greeting_responses = [
        "Hello! I'm SkillBot. How can I help you today?",
        "Hi there! This is SkillBot. What would you like to know?",
        "Greetings! I'm SkillBot, your helpful assistant. How can I assist you?",
        "Hey! I'm SkillBot. Ready to answer your questions!"
    ]
    import random
    return random.choice(greeting_responses)

@app.post("/query", response_model=QueryResponse, tags=["Chat"])
async def query(request: QueryRequest):
    """
    Send a query to the chatbot and get a response
    
    Parameters:
    - query: The question to ask the chatbot
    
    Returns:
    - query: The original query
    - response: The chatbot's response
    """
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Check if the message is a greeting
        if is_greeting(request.query):
            response = get_greeting_response(request.query)
        else:
            response = qa_chain.invoke(request.query)
        
        return QueryResponse(query=request.query, response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    # Run the server with: python api.py
    # Or use: uvicorn api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
